from __future__ import annotations

import csv
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from docx import Document

from docx_support import add_bullets, add_numbered_references, add_table, add_title_page, save_doc, style_document


ROOT = Path(__file__).resolve().parents[1]
OUTCOME_INVENTORY_PATH = ROOT / "output" / "spreadsheet" / "gold_available_outcomes_latest_round.csv"
STEP0_INVENTORY_PATH = ROOT / "artifacts" / "analysis" / "step0" / "frozen_outcome_inventory.csv"
ROOT_DOC_PATH = ROOT / "heat_mnch_structured_analysis_plan_thresholds_vancouver.docx"
OUTPUT_DOC_PATH = ROOT / "output" / "doc" / "heat_mnch_incremental_analysis_plan_v1.docx"
OUTPUT_MD_PATH = ROOT / "output" / "doc" / "heat_mnch_incremental_analysis_plan_v1.md"
ARCHIVE_DIR = ROOT / "output" / "doc" / "archive" / "previous_analysis_plans"

VARIABLE_NORMALIZATION = {
    "STILLBIRTH_TOTAL_PRIMARY_GTE28": {"STILLBIRTH_TOTAL_PRIMARY_GTE28", "STILLBIRTH_PRIMARY_GTE28_FLAG"},
    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": {"STILLBIRTH_TOTAL_GTE22_SENSITIVITY", "STILLBIRTH_SENS_GTE22_FLAG"},
    "DEATHS_0_27D": {"DEATHS_0_27D", "DEATHS_NEONATAL"},
}

SUPPLEMENTAL_IMPLEMENTED_OUTCOME_ROWS = [
    [
        "Child under-5 morbidity and hospitalization",
        "Direct heat hospitalization (SIH grouped)",
        "Descriptive-only",
        "CHILD_DIRECT_HEAT_HOSPITALIZATION_TOTAL",
        "gold_mnch_mortality",
    ],
    [
        "Child under-5 morbidity and hospitalization",
        "Indirect heat-sensitive hospitalization (SIH grouped)",
        "Descriptive-only",
        "CHILD_INDIRECT_HEAT_HOSPITALIZATION_TOTAL",
        "gold_mnch_mortality",
    ],
    [
        "Child under-5 morbidity and hospitalization",
        "Excessive natural heat X30 hospitalization linkage",
        "Descriptive-only",
        "CHILD_EXCESSIVE_NATURAL_HEAT_X30_HOSPITALIZATION_TOTAL",
        "gold_mnch_mortality",
    ],
]

REPORT_STAGE_CONFIG = [
    (
        "Silver release candidate",
        ROOT / "silver_release_candidate_report.md",
        "Silver source integrity, denominator alignment, and analytical daily views are fully gated.",
        "No modelling round should begin until this gate stays green.",
    ),
    (
        "Gold Stage 1 perinatal",
        ROOT / "gold_stage1_perinatal_report.md",
        "Canonical perinatal gold mart is available for 2020-2024.",
        "Primary source for preterm, low birth weight, stillbirth, ANC, and gestational severity endpoints.",
    ),
    (
        "Gold Stage 2 mortality",
        ROOT / "gold_stage2_mortality_report.md",
        "Canonical mortality gold mart is available for 2020-2024.",
        "Primary source for neonatal, infant, under-five, hospitalization, and heat-coded outcomes.",
    ),
    (
        "Gold Stage 3 panel",
        ROOT / "gold_stage3_panel_report.md",
        "Harmonized municipality-day panel is available for 2020-2024.",
        "Backbone for aggregated threshold, lag, and operational monitoring analyses.",
    ),
    (
        "Gold Stage 4 derived",
        ROOT / "gold_stage4_derived_report.md",
        "Derived analytical views are available without rewriting canonical gold assets.",
        "Supports aggregated maternal and child analytical marts for immediate modelling.",
    ),
    (
        "Gold Stage 5 dictionary",
        ROOT / "gold_stage5_dictionary_report.md",
        "Documentation, variable dictionary, and endpoint readiness ledger are current.",
        "Use this layer to lock names, denominators, and readiness status before every modelling round.",
    ),
    (
        "Gold promotion",
        ROOT / "gold_promotion_report.md",
        "The full canonical gold and derived view stack has been promoted successfully.",
        "This is the repository-level confirmation that the current cycle is build-complete.",
    ),
]

DRAFT_PLAN_PATHS = [
    ROOT / "threshold_lag_heterogeneity_structured_analysis_plan_v2.docx",
    ROOT / "threshold_lag_heterogeneity_structured_analysis_plan_v3_updated_outcomes.docx",
    ROOT / "threshold_lag_heterogeneity_structured_analysis_plan_v3_supplement_outcome_crosswalk.docx",
]

GROUP_ROLE_NOTES = {
    "Perinatal and antenatal care": "Immediate modelling with explicit separation of headline perinatal outcomes and supplementary care-process indicators.",
    "Maternal and WRA canonical": "Immediate aggregated modelling in the current cycle.",
    "Maternal and WRA sensitivity": "Sensitivity-only confirmation of maternal findings under the narrower denominator.",
    "Child under-5 mortality": "Neonatal, infant, and under-five mortality advance immediately; child_u5 age-split extensions remain in-plan, but DEATHS_0_27D is treated as a duplicate of DEATHS_NEONATAL rather than as a separate live Step 0 row.",
    "Child under-5 morbidity and hospitalization": "All child hospitalization endpoints enter the plan, including dehydration, diarrhea or GI, respiratory acute, asthma or wheeze, and the SIH-grouped direct, indirect, and X30 heat-coded hospitalizations. Step 1 verifies whether each should stay descriptive-only or be promoted to Step 2.",
    "Heat-related child outcomes": "All heat-coded child outcomes stay in the plan, but they should remain descriptive or sparse surveillance lines until support is demonstrated.",
    "Operational proxy": "Retain for service-pressure triangulation only; never allow it to replace clinically anchored outcomes.",
}

SEASONAL_CONTROL_ROWS = [
    [
        "Acute child mortality, child hospitalization, heat-coded mortality, and operational proxy",
        "Municipality fixed effects plus calendar year, explicit month or seasonal calendar control, and day-of-week for daily models.",
        "Epidemiologic week can be used as a robustness layer when acute timing is central.",
        "Mandatory from baseline fit onward.",
    ],
    [
        "Maternal WRA aggregated count outcomes",
        "Municipality fixed effects plus calendar year and explicit month or seasonal calendar control.",
        "Use day-of-week when the specification remains daily; otherwise use month or season in grouped time designs.",
        "Mandatory from baseline fit onward.",
    ],
    [
        "Perinatal grouped birth outcomes and stillbirth",
        "Calendar year plus month of conception for conception-anchored models, or delivery or event month for delivery-aligned models.",
        "Do not mix conception-month and delivery-month adjustment carelessly; the timing control must match the exposure window definition.",
        "Mandatory from baseline fit onward.",
    ],
    [
        "Neonatal acute analyses",
        "Calendar year, calendar month, and day-of-week in the main acute specification.",
        "Epidemiologic-week conditioning should be carried as a robustness check when acute daily timing is a headline claim.",
        "Mandatory in the escalation stream and strongly preferred in aggregated acute work.",
    ],
    [
        "UF pooling, burden estimation, and final synthesis",
        "No new standalone seasonal term should be introduced here; these stages must inherit first-stage estimates that were already seasonality-adjusted.",
        "Burden and pooled heterogeneity outputs should be rejected if based on seasonally under-adjusted first-stage models.",
        "Mandatory by inheritance.",
    ],
]

SEASONAL_STAGE_ROWS = [
    [
        "Step 0",
        "Indispensable",
        "Freeze the seasonal control contract by outcome family before any model fitting.",
    ],
    [
        "Step 1",
        "Indispensable",
        "Inspect monthly and seasonal outcome support, fallback shares, and exposure distributions before promotion.",
    ],
    [
        "Step 2",
        "Indispensable",
        "Baseline models must already include the required month or season or DOW timing structure for the outcome type.",
    ],
    [
        "Step 3",
        "Indispensable",
        "Threshold discovery must be conditioned on the same seasonal controls, otherwise upper-tail signals may be seasonal confounding.",
    ],
    [
        "Step 4",
        "Indispensable",
        "Lag comparison must keep the seasonal control set fixed while the lag window changes.",
    ],
    [
        "Step 5",
        "Indispensable by inheritance",
        "UF pooling and municipal susceptibility synthesis must use first-stage estimates that were already seasonality-adjusted.",
    ],
    [
        "Step 6",
        "Indispensable by inheritance",
        "Absolute burden estimates are acceptable only if the upstream preferred model already passed the seasonal-adjustment gate.",
    ],
    [
        "Step 7",
        "Indispensable",
        "Event-level escalation models must include conception month, event month, DOW, or epidemiologic week as appropriate to the design.",
    ],
    [
        "Step 8",
        "Mandatory post-analytic disclosure",
        "The editorial closeout must lock claim tiers, temporal controls, robustness checks, and allowed external language before any round summary is shared.",
    ],
    [
        "Step 9",
        "Required for next-cycle readiness",
        "The next-cycle agenda must convert unresolved model, validation, and reporting gaps into prioritized checklist items with dependencies and done-when criteria.",
    ],
]

STEP_CONFIG = [
    {
        "step": "Step 0",
        "title": "Freeze, supersession, and governance reset",
        "paragraphs": [
            "This plan replaces the earlier Step A/Step B framing with a single incremental programme. The governing distinction is no longer an abstract two-track split, but a practical escalation order: first use what the validated current-cycle aggregated stack can already answer, then open an event-level escalation round only for outcomes that justify deeper timing or sparse-outcome work.",
            "At this step the team locks the current gold cycle, the outcome inventory, denominator rules, and the naming conventions that will feed reports and dashboards. No new modelling should start until these governance objects are frozen for the round.",
            "The seasonal adjustment contract also has to be frozen here. In the five-year round, seasonality is not optional decoration: month, season, day-of-week, and epidemiologic-week controls must be assigned explicitly by outcome family before modelling begins.",
        ],
        "deliverables": [
            "Frozen outcome inventory with primary versus sensitivity labels.",
            "CSV-to-live inventory reconciliation table proving that every CSV-listed variable has an explicit route.",
            "Locked denominator assignments and exposure-family defaults by outcome family.",
            "Locked seasonal control contract by outcome family, stating whether the main control is month, season, day-of-week, conception month, event month, or epidemiologic-week robustness.",
            "Round charter stating which outputs are immediate, supplementary, or deferred.",
        ],
        "gate": [
            "All silver-to-gold reports remain PASS for 2020-2024.",
            "Every row in gold_available_outcomes_latest_round.csv is either mapped into live Step 0, matched through an approved equivalence, or explicitly blocked and explained.",
            "The outcome roster for the round is approved in writing.",
            "The seasonal control contract is approved before any model is fit.",
            "Any dashboard language changes stay presentation-only and do not alter API ids or file contracts.",
        ],
        "if_blocked": "If any freeze item is unresolved, stop here and fix metadata, denominators, or governance language before any model fitting.",
    },
    {
        "step": "Step 1",
        "title": "Descriptive readiness atlas and denominator audit",
        "paragraphs": [
            "The first analytical step is descriptive, not inferential. It should profile event support, exposure tails, missingness, fallback climate shares, denominator completeness, and geographic concentration by outcome family and by year.",
            "This step exists to prevent over-ambitious modelling. Every later model should inherit a documented descriptive support statement from this readiness atlas.",
            "This is also the first mandatory seasonality screen. Monthly and seasonal distributions of outcomes and exposures should be inspected before any threshold or lag claim is allowed to proceed, because a five-year series can still be badly distorted by unresolved seasonal structure.",
        ],
        "deliverables": [
            "Outcome-family descriptive dossier with counts, rates, exposure percentiles, and missingness.",
            "Monthly and seasonal descriptive atlas for outcomes, denominators, and climate exposures.",
            "Denominator integrity checks for live births, total births, under-five population, and women-of-reproductive-age populations.",
            "Sparse-outcome flags for heat-coded mortality and small age-band strata.",
        ],
        "gate": [
            "Primary endpoints have adequate event support for at least a baseline model.",
            "Monthly and seasonal profiles have been inspected and documented before outcome promotion.",
            "Denominator arithmetic is internally consistent for every promoted outcome.",
            "Climate fallback reliance is quantified and acceptable for the intended main analysis.",
        ],
        "if_blocked": "If descriptive support is weak, downgrade the outcome to sensitivity-only or descriptive-only before moving forward.",
    },
    {
        "step": "Step 2",
        "title": "Baseline aggregated models by outcome family",
        "paragraphs": [
            "Step 2 fits the simplest defensible model family for each outcome structure using the current aggregated stack. Count outcomes start with Poisson or quasi-Poisson municipality fixed-effects models, with rapid escalation to negative binomial where overdispersion requires it. Birth-event outcomes use grouped binomial or Poisson-with-offset representations depending on the delivered mart.",
            "From this step onward, every promoted candidate should be run in parallel under maximum temperature and approximate heat index using matched baseline windows. Neither exposure family should become the headline family yet; the purpose here is to establish whether both families are estimable, directionally coherent, and ready for fair comparison.",
            "This step is not yet a threshold paper. Its purpose is to establish that the direction, scale, and denominator behavior of the core outcomes are stable enough to deserve a more flexible exposure-response search.",
            "Seasonality enters explicitly at this stage and is indispensable. Daily acute outcomes should carry day-of-week plus month or seasonal calendar controls; maternal and child aggregated daily counts should retain year plus month or season; conception-anchored perinatal models should carry conception month; delivery-aligned or stillbirth models should carry delivery or event month; neonatal acute analyses should additionally preserve epidemiologic-week robustness as a declared sensitivity.",
        ],
        "deliverables": [
            "Baseline model table for each promoted outcome family.",
            "Model specification matrix listing the mandatory temporal controls used for each outcome family.",
            "Overdispersion and fit diagnostics with explicit escalation decisions.",
            "Primary and sensitivity denominator confirmation notes.",
            "Exposure-family comparison table showing whether tmax and heat index are both estimable for each promoted candidate.",
        ],
        "gate": [
            "At least one defensible baseline specification converges for each promoted outcome.",
            "The required temporal controls are present and correctly aligned with the outcome and exposure timing definition.",
            "Primary outcomes are not driven by denominator errors or unstable offsets.",
            "Sensitivity denominator choices do not reverse the main finding direction without explanation.",
        ],
        "if_blocked": "If the baseline model is unstable, the outcome does not proceed to threshold discovery in the current round.",
    },
    {
        "step": "Step 3",
        "title": "Threshold discovery and exposure-family choice",
        "paragraphs": [
            "Threshold discovery begins only after a stable baseline exists. Maximum temperature and heat index should be treated as parallel exposure families, screened separately. Flexible upper-tail screens come first; segmented or piecewise models then estimate candidate trigger points instead of imposing a single national cut-off.",
            "The key decision at this step is not only whether a threshold exists, but whether the threshold signal is stable enough to justify the extra interpretive weight that comes with change-point language.",
            "The seasonal control set from Step 2 must remain in place during threshold discovery. In a five-year series, omitted month or season structure can easily masquerade as a false threshold, especially for birth outcomes and acute outcomes with strong annual cycles.",
        ],
        "deliverables": [
            "Threshold candidate registry for maximum temperature and heat index by outcome family.",
            "Candidate trigger diagnostics using information criteria or profile-likelihood style comparisons.",
            "Recommended main exposure family for each promoted outcome.",
        ],
        "gate": [
            "The preferred exposure family is selected separately for each outcome family.",
            "Threshold candidates remain directionally stable after explicit seasonal adjustment.",
            "The threshold estimate is directionally stable across reasonable candidate specifications.",
            "No headline threshold claim is made for outcomes with flat or highly unstable upper-tail behavior.",
        ],
        "if_blocked": "If no stable threshold candidate emerges, keep the outcome in linear or descriptive mode and do not force a threshold narrative.",
    },
    {
        "step": "Step 4",
        "title": "Lag escalation and timing refinement",
        "paragraphs": [
            "Only after a threshold-capable exposure family is chosen should lag escalation begin. Acute outcomes should start with short windows such as lag 0, lag 0-1, lag 0-3, and lag 0-7; gestational outcomes should start with interpretable delivery-aligned or late-pregnancy windows that already exist in the stack.",
            "Distributed lag surfaces are promoted only where the event support and numerical stability justify them. The current cycle should avoid presenting delivery-aligned aggregates as if they were full trimester- or whole-pregnancy causal windows.",
            "This step should change lag structure while keeping the temporal adjustment contract fixed. Otherwise a supposed lag improvement may just reflect inconsistent handling of month, season, or day-of-week controls across candidate models.",
        ],
        "deliverables": [
            "Lag comparison table for each promoted outcome.",
            "Timing narrative stating what the selected lag/window can and cannot mean biologically.",
            "Record of which seasonal controls were held fixed across the lag comparison ladder.",
            "Promotion list of outcomes that deserve distributed-lag or event-level timing work in a later round.",
        ],
        "gate": [
            "The preferred lag/window is supported by better fit or clearer stability than the simpler alternatives.",
            "Lag comparisons were run under a fixed seasonal adjustment structure rather than under changing temporal-control rules.",
            "Sparse outcomes are not over-parameterized with flexible lag surfaces.",
            "Timing claims remain aligned with the observed exposure window actually available in the data.",
        ],
        "if_blocked": "If lag refinement is unstable, fall back to the simpler timing window and keep the interpretation narrow.",
    },
    {
        "step": "Step 5",
        "title": "Heterogeneity and municipality susceptibility synthesis",
        "paragraphs": [
            "Once the first-stage effect is stable, the plan shifts to heterogeneity. Municipality-level estimates should be summarized and then pooled at the UF level so that susceptibility patterns are described in a way that is both statistically stable and institutionally aligned with the SUS.",
            "Contextual variables belong here, after the local effect is estimated. They explain heterogeneity; they do not replace municipality fixed effects in the main design.",
            "Seasonality is not reintroduced here as a new optional covariate. Instead, this stage inherits first-stage estimates that should already be seasonality-adjusted. If the first-stage model omitted required temporal controls, the heterogeneity stage should be treated as blocked.",
        ],
        "deliverables": [
            "Municipality and UF pooled summaries of threshold or slope behavior.",
            "Contextual effect-modification tables for urbanicity, inequality, development, altitude, or coastal status where supported.",
            "Susceptibility profile classification separating stable activation from null or fragile signals.",
        ],
        "gate": [
            "First-stage estimates exist for enough municipalities to justify pooling.",
            "Effect-modification claims are not made before first-stage stability is documented.",
            "Contextual contrasts are presented as heterogeneity evidence rather than universal adjusted coefficients.",
        ],
        "if_blocked": "If municipal estimates are too sparse or noisy, report only descriptive geographic variation and defer pooled heterogeneity claims.",
    },
    {
        "step": "Step 6",
        "title": "Absolute burden and policy translation",
        "paragraphs": [
            "Burden estimation is a late-stage privilege, not a default output. It should be produced only for outcomes that already passed the stability gates for baseline fit, threshold behavior, and timing.",
            "For this project the policy value of the analysis depends on translating relative effects into absolute excess counts or risk differences where numerically defensible. Burden estimates should therefore be tightly limited to the most stable outcomes.",
            "No burden output should be accepted from a model that did not already satisfy the seasonal-adjustment requirements upstream. Seasonality should therefore be treated as an inherited gate here, not as a late add-on.",
        ],
        "deliverables": [
            "Absolute excess count or risk-difference summaries for the strongest outcomes.",
            "Short policy translation notes stating who should use the estimate and under which caution labels.",
            "Explicit exclusions list for outcomes not eligible for burden work.",
        ],
        "gate": [
            "Relative effects and threshold estimates are both stable.",
            "The additive estimate is numerically well behaved or can be derived transparently from the preferred multiplicative model.",
            "Burden estimates are not dominated by a handful of municipalities or rare strata.",
        ],
        "if_blocked": "If stability is weak, keep the result relative-only and document why burden translation was not approved.",
    },
    {
        "step": "Step 7",
        "title": "Event-level escalation round",
        "paragraphs": [
            "A separate escalation round should be opened only for questions the aggregated current-cycle stack cannot answer well. This includes richer gestational timing for preterm and stillbirth, deeper neonatal acute timing, and any sparse-outcome framework that truly needs partial pooling or specialized event-level handling.",
            "This round should never be treated as routine spillover. It is a deliberate new round with its own success criteria, because it increases computational complexity, interpretive risk, and operational burden.",
            "When this round is opened, explicit temporal controls become even more important. Event-level or grouped event-level models should carry conception month, event month, day-of-week, and epidemiologic-week structures as appropriate to the question rather than leaving seasonality implicit.",
        ],
        "deliverables": [
            "Event-level or grouped event-level matrix refresh for priority perinatal outcomes.",
            "Escalation memo stating why the aggregated cycle was insufficient for each promoted question.",
            "Temporal-control appendix showing which month, event-month, day-of-week, or epidemiologic-week controls were used in the escalation model.",
            "Separate results bundle clearly labeled as escalation-round evidence.",
        ],
        "gate": [
            "The outcome failed a current-cycle limitation that event-level data can plausibly fix.",
            "The question is important enough to justify added computational and governance cost.",
            "The event-level model has explicit temporal controls matched to the outcome process before any escalation result is promoted.",
            "The escalation design has explicit stopping rules if the expected gain does not materialize.",
        ],
        "if_blocked": "If the escalation rationale is weak, keep the outcome in the aggregated current-cycle package and do not open a new round.",
    },
    {
        "step": "Step 8",
        "title": "Post-analytic editorial closeout and interim translation",
        "paragraphs": [
            "After Step 7, the analytical round is closed computationally. Step 8 is not a new executable modelling step. It is the post-analytic editorial layer that freezes how the round may be described, translated, and partially shared.",
            "This step exists because the current stack can produce more signals than should be elevated into headline language. The closeout therefore has to classify which findings are headline-ready, which remain supportive, which stay exploratory, which are not externally reportable, and which must be deferred to a future cycle.",
            "The disclosure layer is mandatory and logically prior to any institutional note, manuscript paragraph, or dashboard wording. Readers should never have to infer which temporal controls were used, whether the evidence is inherited aggregated versus event-level escalated, or whether a signal is being presented as burden-ready versus relative-only.",
        ],
        "deliverables": [
            "Claims-and-disclosure appendix locking claim tiers, source artifacts, required caveats, and allowed external language.",
            "Scientific master synthesis for the current round, derived from the claims register rather than from ad hoc narrative.",
            "Interim OMS/WMO technical note derived only from claims explicitly marked as externally reportable.",
        ],
        "gate": [
            "The Step 7 snapshot used for editorial closeout is frozen and named explicitly.",
            "Every promoted outcome family is classified as headline, supportive, exploratory, not externally reportable, or deferred.",
            "Every claim in the master synthesis and interim note maps to a documented upstream artifact and a required caveat.",
            "User-facing language remains downstream of the evidence class and does not create a new analytical surface by narrative.",
        ],
        "if_blocked": "If claim governance is incomplete, do not let scattered model outputs leak directly into manuscript, dashboard, or institutional narratives.",
    },
    {
        "step": "Step 9",
        "title": "Next-cycle readiness and upgrade agenda",
        "paragraphs": [
            "Step 9 is the readiness surface for the next cycle. It is still part of the editorial closeout rather than the executable pipeline, and its purpose is operational: convert everything the current round could not resolve into a dependency-aware checklist for the next modelling and publication cycle.",
            "This step should not read like a vague roadmap. Every item must state the problem it addresses, the dependency required to solve it, what done looks like, and how it changes the next round. The goal is to stop the team from reopening the analytical cycle without a clear upgrade agenda.",
        ],
        "deliverables": [
            "Next-cycle readiness checklist with must, should, and nice priority classes.",
            "Upgrade agenda covering model design, burden reconciliation, validation, heterogeneity, and publication governance.",
            "Explicit publication-readiness rules stating what must be fixed before stronger external claims are allowed.",
        ],
        "gate": [
            "Every major unresolved claim, model, validation, or reporting issue from the current round is converted into a checklist item.",
            "Each checklist item names a dependency and a concrete done-when criterion.",
            "The next round is not opened until the readiness checklist is reviewed and accepted.",
        ],
        "if_blocked": "If the next-cycle agenda is still vague, do not promise stronger claims or open a new modelling round.",
    },
]

RECOMMENDATION_ROWS = [
    [
        "Improvement",
        "Treat gold_available_outcomes_latest_round.csv as the authoritative round inventory and reconcile the live Step 0 roster against it before Step 2.",
        "No CSV-listed variable should silently disappear from the plan; each one needs an explicit route, even if the route is an approved equivalence, defer, cut, or Step 0 expansion.",
    ],
    [
        "Improvement",
        "Make explicit seasonal adjustment mandatory in the five-year round: month or season plus day-of-week or epidemiologic-week robustness when the outcome process is acute.",
        "With only 2020-2024 in scope, unresolved seasonal structure can easily be mistaken for a heat threshold or lag signal.",
    ],
    [
        "Improvement",
        "Freeze one temporal-control contract per outcome family before fitting any model.",
        "The team should not allow threshold and lag decisions to drift because month, season, or DOW handling changed between candidate models.",
    ],
    [
        "Improvement",
        "Run tmax and approximate heat index in parallel from Step 2 onward and defer the headline exposure-family choice until Step 3 or Step 4.",
        "This preserves transparency and comparability while letting heat index win only when it shows equal or better support without weaker coherence.",
    ],
    [
        "Improvement",
        "Add a threshold candidate registry to every promoted outcome family.",
        "Threshold claims will otherwise drift across notebooks and be hard to compare across municipalities and UFs.",
    ],
    [
        "Improvement",
        "Emit a short decision memo after every modelling step.",
        "The team needs a durable reason for every promotion, cut, or defer decision.",
    ],
    [
        "Improvement",
        "Expose event support, fallback climate share, and denominator status in dashboard-facing summaries.",
        "These are the core operational qualifiers for reading the estimates correctly.",
    ],
    [
        "Cut",
        "Keep ANC 8+ contacts out of the primary causal headline package.",
        "It is more defensible as a supplementary care-process or mediator-sensitive endpoint.",
    ],
    [
        "Cut",
        "Keep any-mention heat-coded mortality and asthma/wheeze out of the main inferential narrative unless support improves.",
        "These endpoints are too vulnerable to sparse cells or coding heterogeneity for headline claims right now.",
    ],
    [
        "Cut",
        "Do not present delivery-aligned aggregates as trimester or whole-pregnancy causal effects in the current cycle.",
        "The current gold stack does not warrant that timing claim without an explicit event-level escalation round.",
    ],
    [
        "New round",
        "Open a perinatal event-level escalation round for preterm, stillbirth, and neonatal timing only after aggregated screening prioritizes them.",
        "This is the most credible route to stronger gestational timing inference.",
    ],
    [
        "New round",
        "Run UF-pooled threshold meta-analysis only after municipal first-stage models stabilize.",
        "Pooling too early will blur rather than clarify heterogeneity.",
    ],
    [
        "New round",
        "Produce attributable burden outputs only for outcomes that pass all stability gates.",
        "This protects policy translation from over-interpretation.",
    ],
]

MASTER_CHECKLIST = [
    "Silver and gold gates remain PASS for all required years before the round begins.",
    "Every row in gold_available_outcomes_latest_round.csv is either mapped into the live Step 0 roster, matched through an approved equivalence, or explicitly blocked and explained.",
    "The current round explicitly names which outcomes are headline, supplementary, and deferred.",
    "Primary versus sensitivity denominator rules are frozen before modelling.",
    "The seasonal control contract is frozen before modelling and maps each outcome family to month, season, DOW, conception month, event month, or epidemiologic-week robustness as appropriate.",
    "Maximum temperature and heat index are considered separately before a main exposure family is chosen.",
    "No outcome advances without a stable baseline model.",
    "No outcome advances without explicit seasonal adjustment appropriate to its outcome process.",
    "No threshold claim advances without candidate-stability evidence.",
    "No threshold or lag comparison changes the seasonal control structure while comparing exposure specifications.",
    "No lag escalation advances if a simpler window is already unstable.",
    "No heterogeneity claim advances before first-stage estimates are stable.",
    "No burden estimate advances before the relative model is mature.",
    "Sparse heat-coded or any-mention outcomes are clearly labeled as descriptive or sensitivity-only when appropriate.",
    "Dashboard wording is updated without changing API ids, manifest paths, or storage contracts.",
    "The next round is defined only after the current synthesis identifies what should be improved, cut, or escalated.",
]

REFERENCE_LIST = [
    "World Health Organization. Births, preterm, rate per 100 live births.",
    "World Health Organization. Stillbirth rate per 1000 total births.",
    "World Health Organization. Neonatal, infant, and under-five mortality indicator metadata.",
    "Brimicombe C, Conway F, Portela A, et al. A scoping review on heat indices used to measure the effects of heat on maternal and perinatal health. BMJ Public Health. 2024;2(1):e000308.",
    "Lakhoo DP, Brink N, Radebe L, et al. A systematic review and meta-analysis of heat exposure impacts on maternal, fetal and neonatal health. Nat Med. 2025;31(2):684-694.",
    "Gasparrini A, Armstrong B, Kenward MG. Distributed lag non-linear models. Stat Med. 2010;29(21):2224-2234.",
    "Muggeo VMR. Analyzing temperature effects on mortality within the R environment: the constrained segmented distributed lag parameterization. J Stat Softw. 2010;32(12):1-17.",
    "Gasparrini A, Leone M. Attributable risk from distributed lag models. BMC Med Res Methodol. 2014;14:55.",
    "Hougaard P. Frailty models for survival data. Lifetime Data Anal. 1995;1(3):255-273.",
    "Balan TA, Putter H. A tutorial on frailty models. Stat Methods Med Res. 2020;29(11):3424-3454.",
]


def _report_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not path.exists():
        return fields
    pattern = re.compile(r"^- ([^:]+): `(.*)`$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw_line.strip())
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


def _load_outcome_inventory() -> list[dict[str, str]]:
    with OUTCOME_INVENTORY_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_step0_inventory() -> list[dict[str, str]]:
    if not STEP0_INVENTORY_PATH.exists():
        return []
    with STEP0_INVENTORY_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_summary_rows(outcomes: list[dict[str, str]]) -> list[list[str]]:
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tables_by_group: dict[str, set[str]] = defaultdict(set)
    for row in outcomes:
        group = row["group"]
        status_counts[group][row["status"]] += 1
        tables_by_group[group].add(row["gold_table"])
    rows: list[list[str]] = []
    for group in GROUP_ROLE_NOTES:
        rows.append(
            [
                group,
                str(status_counts[group]["Primary"]),
                str(status_counts[group]["Sensitivity"]),
                ", ".join(sorted(tables_by_group[group])),
                GROUP_ROLE_NOTES[group],
            ]
        )
    return rows


def _crosswalk_rows(outcomes: list[dict[str, str]]) -> list[list[str]]:
    rows = [
        [row["group"], row["outcome"], row["status"], row["gold_variable"], row["gold_table"]]
        for row in outcomes
    ]
    rows.extend(SUPPLEMENTAL_IMPLEMENTED_OUTCOME_ROWS)
    return rows


def _equivalent_variables(gold_variable: str) -> set[str]:
    return set(VARIABLE_NORMALIZATION.get(gold_variable, {gold_variable}))


def _inventory_reconciliation_rows(
    outcomes: list[dict[str, str]],
    step0_inventory: list[dict[str, str]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for outcome in outcomes:
        gold_variable = outcome["gold_variable"]
        equivalents = _equivalent_variables(gold_variable)
        matches = [
            row
            for row in step0_inventory
            if row.get("canonical_variable", "") in equivalents
        ]
        roles = sorted({row.get("analysis_role", "") for row in matches if row.get("analysis_role")})
        normalized_names = sorted(
            {
                row.get("canonical_variable", "")
                for row in matches
                if row.get("canonical_variable", "") and row.get("canonical_variable", "") != gold_variable
            }
        )

        if matches:
            live_step0 = "Yes"
            if any(role in {"primary", "sensitivity"} for role in roles):
                planned_route = "Immediate or sensitivity round"
                verification_path = "Step 1 to Step 6; Step 7 only if escalation is needed"
            elif "descriptive_only" in roles:
                planned_route = "Descriptive-only gate"
                verification_path = "Step 1 descriptive atlas first; promote only after governance reset"
            elif "deferred_event_level" in roles:
                planned_route = "Escalation-only"
                verification_path = "Step 7 after aggregated screening"
            else:
                planned_route = "Needs governance review"
                verification_path = "Step 0 reconciliation"
            note_parts = [f"Live Step 0 role(s): {', '.join(roles)}"] if roles else ["Matched in live Step 0."]
            if normalized_names:
                note_parts.append(f"Normalized live name(s): {', '.join(normalized_names)}")
            if gold_variable == "DEATHS_0_27D" and "DEATHS_NEONATAL" in normalized_names:
                note_parts.append(
                    "Approved equivalence: this CSV row is covered by the neonatal deaths line and does not require a separate Step 0 outcome."
                )
            notes = " ".join(note_parts)
        else:
            live_step0 = "No"
            planned_route = "Planned but not yet in live Step 0"
            verification_path = "Step 0 inventory expansion before Step 1 or Step 2 use"
            notes = "Present in the CSV master inventory but absent from the current live Step 0 roster."

        rows.append(
            [
                outcome["group"],
                outcome["outcome"],
                outcome["status"],
                gold_variable,
                outcome["gold_table"],
                live_step0,
                planned_route,
                verification_path,
                notes,
            ]
        )
    return rows


def _inventory_reconciliation_summary_rows(reconciliation_rows: list[list[str]]) -> list[list[str]]:
    live_yes = sum(1 for row in reconciliation_rows if row[5] == "Yes")
    live_no = sum(1 for row in reconciliation_rows if row[5] == "No")
    descriptive_gate = sum(1 for row in reconciliation_rows if row[6] == "Descriptive-only gate")
    immediate_gate = sum(1 for row in reconciliation_rows if row[6] == "Immediate or sensitivity round")
    normalized = sum(1 for row in reconciliation_rows if "Normalized live name(s):" in row[8])
    return [
        ["CSV inventory rows", str(len(reconciliation_rows)), "Every row in gold_available_outcomes_latest_round.csv enters the plan."],
        ["Conceptually represented in live Step 0", str(live_yes), "These variables already have a live mapping in the current Step 0 roster."],
        ["Blocked pending Step 0 inventory expansion", str(live_no), "These variables remain in-plan, but they are not yet represented in the live Step 0 roster."],
        ["Variables currently gated as descriptive-only", str(descriptive_gate), "These enter Step 1 and require explicit promotion before Step 2 modelling."],
        ["Variables already on immediate or sensitivity routes", str(immediate_gate), "These can proceed through Step 1 and into later modelling steps if they pass gates."],
        ["Variables matched through approved equivalence", str(normalized), "Stillbirth rows and the neonatal duplicate row can be carried through normalized live Step 0 names."],
    ]


def _stage_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for stage_name, path, current_state, plan_role in REPORT_STAGE_CONFIG:
        fields = _report_fields(path)
        rows.append(
            [
                stage_name,
                fields.get("Overall status", "UNKNOWN"),
                current_state,
                plan_role,
            ]
        )
    return rows


def _roadmap_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for step in STEP_CONFIG:
        rows.append(
            [
                f"{step['step']} - {step['title']}",
                step["deliverables"][0],
                step["gate"][0],
                step["if_blocked"],
            ]
        )
    return rows


def _summary_metrics(outcomes: list[dict[str, str]]) -> dict[str, str]:
    total = len(outcomes)
    primary = sum(1 for row in outcomes if row["status"] == "Primary")
    sensitivity = sum(1 for row in outcomes if row["status"] == "Sensitivity")
    groups = len({row["group"] for row in outcomes})
    tables = len({row["gold_table"] for row in outcomes})
    stage5_fields = _report_fields(ROOT / "gold_stage5_dictionary_report.md")
    return {
        "total_outcomes": str(total),
        "primary": str(primary),
        "sensitivity": str(sensitivity),
        "groups": str(groups),
        "tables": str(tables),
        "ready_now_endpoints": stage5_fields.get("Ready now endpoints", "NA"),
        "dictionary_rows": stage5_fields.get("Variable dictionary rows", "NA"),
    }


def _archive_path(path: Path, suffix_label: str) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return ARCHIVE_DIR / f"{path.stem}_{suffix_label}_{stamp}{path.suffix}"


def _archive_existing_root_doc() -> None:
    if ROOT_DOC_PATH.exists():
        shutil.copy2(ROOT_DOC_PATH, _archive_path(ROOT_DOC_PATH, "pre_refresh"))


def _archive_superseded_drafts() -> list[str]:
    moved: list[str] = []
    for path in DRAFT_PLAN_PATHS:
        if not path.exists():
            continue
        destination = _archive_path(path, "archived")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        moved.append(str(destination.relative_to(ROOT)))
    return moved


def _cleanup_hidden_clutter() -> list[str]:
    removed: list[str] = []
    for pattern in (".DS_Store", ".Rhistory"):
        for path in ROOT.rglob(pattern):
            if ".git" in path.parts:
                continue
            path.unlink(missing_ok=True)
            removed.append(str(path.relative_to(ROOT)))
    return sorted(removed)


def _append_markdown_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        safe_row = [cell.replace("\n", "<br>") for cell in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    lines.append("")


def _build_markdown(
    *,
    metrics: dict[str, str],
    moved_drafts: list[str],
    removed_clutter: list[str],
    group_rows: list[list[str]],
    stage_rows: list[list[str]],
    roadmap_rows: list[list[str]],
    crosswalk_rows: list[list[str]],
    reconciliation_summary_rows: list[list[str]],
    reconciliation_rows: list[list[str]],
) -> str:
    lines: list[str] = []
    lines.append("# HEAT-MNCH Incremental Analysis Plan")
    lines.append("")
    lines.append("## Superseding note")
    lines.append("")
    lines.append(
        "This plan supersedes the earlier Step A/Step B framing. The new structure uses a single incremental escalation path: "
        "start with the validated aggregated current-cycle stack, then open an event-level escalation round only for questions "
        "the current stack cannot answer credibly."
    )
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        f"The repository is currently green through Silver RC, Gold Stages 1 to 5, and gold promotion for 2020-2024. "
        f"The available outcome inventory contains {metrics['total_outcomes']} outcome rows across {metrics['groups']} substantive groups "
        f"and {metrics['tables']} gold tables, with {metrics['primary']} primary and {metrics['sensitivity']} sensitivity entries. "
        f"Stage 5 currently certifies {metrics['ready_now_endpoints']} ready-now endpoints and {metrics['dictionary_rows']} dictionary rows."
    )
    lines.append("")
    lines.append(
        "The immediate priority is therefore not another data-build sprint. The immediate priority is a disciplined modelling sequence "
        "that decides what should be advanced, what should be cut from the headline package, and what truly requires a new round."
    )
    lines.append("")
    lines.append(
        "For this refreshed plan, gold_available_outcomes_latest_round.csv is treated as the authoritative round inventory. "
        "Every CSV row must enter the plan, even if its current route is descriptive-only, deferred, matched through an approved equivalence, or blocked pending Step 0 reconciliation."
    )
    lines.append("")
    lines.append(
        "The implemented live roster now also carries three supplementary SIH-grouped child hospitalization rows beyond the original CSV master list: "
        "direct heat hospitalization, indirect heat-sensitive hospitalization, and X30-linked hospitalization. These supplementary rows are in-plan for Step 1 to Step 3 screening even though they are not separate original CSV entries."
    )
    lines.append("")
    lines.append("## Official operating mode")
    lines.append("")
    lines.append(
        "The official analytical flow is now steps-only. Operational work should use step-0-freeze through step-6-absolute-burden and steps-run only."
    )
    lines.append("")
    lines.append(
        "Legacy track_a and track_b code remains archived on disk for historical reference, but it is no longer part of the official CLI, the official test suite, or the active execution path."
    )
    lines.append("")
    lines.append(
        "The official internal mart root for the active pipeline is artifacts/analysis/shared_marts. Current and future step outputs should be read from that neutral root rather than from any legacy track-specific artifact directory."
    )
    lines.append("")
    lines.append("## Current stack aligned to this plan")
    lines.append("")
    _append_markdown_table(
        lines,
        ["Layer", "Status", "Current state", "Why it matters"],
        stage_rows,
    )
    lines.append("## Outcome architecture available now")
    lines.append("")
    _append_markdown_table(
        lines,
        ["Group", "Primary", "Sensitivity", "Gold table(s)", "Immediate role in the plan"],
        group_rows,
    )
    lines.append("## CSV master inventory rule and live Step 0 coverage")
    lines.append("")
    lines.append(
        "The current plan is governed by the CSV master inventory, not by the current live Step 0 roster alone. "
        "If a variable appears in gold_available_outcomes_latest_round.csv, it must be carried in the plan with an explicit route."
    )
    lines.append("")
    lines.append(
        "Entry into the plan does not force every variable into immediate modelling. Some variables are immediate, some are sensitivity-only, "
        "some stay descriptive-only until Step 1 promotes them, and some are represented through approved equivalence rather than a duplicate live Step 0 row."
    )
    lines.append("")
    _append_markdown_table(
        lines,
        ["Coverage item", "Count", "Meaning"],
        reconciliation_summary_rows,
    )
    lines.append("## Explicit seasonality contract for the five-year round")
    lines.append("")
    lines.append(
        "For the 2020-2024 round, seasonality must be explicit in the modelling design. The project should not rely on informal temporal balancing. "
        "Instead, the minimum temporal control set must be chosen by outcome process and then held fixed while threshold and lag choices are evaluated."
    )
    lines.append("")
    _append_markdown_table(
        lines,
        ["Outcome family", "Minimum temporal control", "Additional temporal robustness", "Status in this round"],
        SEASONAL_CONTROL_ROWS,
    )
    lines.append("## Where seasonality is indispensable")
    lines.append("")
    _append_markdown_table(
        lines,
        ["Stage", "Status", "Why it matters"],
        SEASONAL_STAGE_ROWS,
    )
    lines.append("## Incremental roadmap")
    lines.append("")
    _append_markdown_table(
        lines,
        ["Step", "Main output", "Advance gate", "If blocked"],
        roadmap_rows,
    )
    for step in STEP_CONFIG:
        lines.append(f"### {step['step']}. {step['title']}")
        lines.append("")
        for paragraph in step["paragraphs"]:
            lines.append(paragraph)
            lines.append("")
        lines.append("Deliverables")
        lines.append("")
        for item in step["deliverables"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Decision checklist")
        lines.append("")
        for item in step["gate"]:
            lines.append(f"- [ ] {item}")
        lines.append("")
        lines.append(f"If blocked: {step['if_blocked']}")
        lines.append("")
    lines.append("## Synthesis")
    lines.append("")
    synthesis_points = [
        "Promote headline work first for primary perinatal outcomes, maternal canonical outcomes, neonatal or infant mortality, under-five mortality, and the three main child hospitalization outcomes.",
        "Carry the SIH-grouped child direct-heat, indirect-heat, and X30-linked hospitalization totals explicitly in the hospitalization block; allow promotion only when support is adequate, and expect the X30-linked hospitalization outcome to stay sparse unless stronger signal appears.",
        "Run maximum temperature and heat index in parallel from Step 2 onward and choose a headline family only after the Step 3 to Step 4 comparison.",
        "Retain very preterm, preterm with low birth weight, stillbirth >=22 weeks, women aged 15-45 sensitivity denominators, asthma or wheeze hospitalization, and direct or indirect heat any-mention outcomes as sensitivity layers unless they show unusually strong and stable support.",
        "Keep ANC 8+ contacts and the acute-care proxy in the programme, but treat them as supplementary operational indicators rather than headline etiologic results.",
        "Keep the age-split child_u5 mortality rows and all heat-coded child mortality rows in the plan even when they are not yet in the live Step 0 roster; their near-term route is Step 0 expansion plus Step 1 verification, not silent removal.",
        "Treat rare heat-coded child mortality as descriptive or sparse-model surveillance unless event support becomes clearly adequate for stable threshold work.",
        "Use the event-level escalation round only for questions that genuinely need richer gestational timing or specialized sparse-outcome handling.",
    ]
    for item in synthesis_points:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    _append_markdown_table(lines, ["Type", "Action", "Rationale"], RECOMMENDATION_ROWS)
    lines.append("## Master decision checklist before the next step")
    lines.append("")
    for item in MASTER_CHECKLIST:
        lines.append(f"- [ ] {item}")
    lines.append("")
    lines.append("## Safe cleanup applied with compatibility protection")
    lines.append("")
    lines.append(
        "The cleanup pass deliberately avoided canonical gold tables, manifest directories, statcalc storage, and API ids. "
        "Only superseded plan drafts and obvious disposable clutter were removed or archived."
    )
    lines.append("")
    if moved_drafts:
        lines.append("Archived prior draft plan files")
        lines.append("")
        for item in moved_drafts:
            lines.append(f"- {item}")
        lines.append("")
    if removed_clutter:
        lines.append("Removed disposable clutter")
        lines.append("")
        for item in removed_clutter:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## References")
    lines.append("")
    for index, ref in enumerate(REFERENCE_LIST, start=1):
        lines.append(f"{index}. {ref}")
    lines.append("")
    lines.append("## Appendix A. Technical crosswalk of the current outcome inventory")
    lines.append("")
    _append_markdown_table(
        lines,
        ["Group", "Outcome", "Status", "Gold variable", "Gold table"],
        crosswalk_rows,
    )
    lines.append("## Appendix B. CSV master inventory reconciliation against the live Step 0 roster")
    lines.append("")
    _append_markdown_table(
        lines,
        ["Group", "Outcome", "CSV status", "Gold variable", "Gold table", "In live Step 0", "Planned route", "Verification path", "Notes"],
        reconciliation_rows,
    )
    return "\n".join(lines).strip() + "\n"


def _build_docx(
    *,
    metrics: dict[str, str],
    moved_drafts: list[str],
    removed_clutter: list[str],
    group_rows: list[list[str]],
    stage_rows: list[list[str]],
    roadmap_rows: list[list[str]],
    crosswalk_rows: list[list[str]],
    reconciliation_summary_rows: list[list[str]],
    reconciliation_rows: list[list[str]],
) -> Document:
    doc = Document()
    style_document(doc)
    add_title_page(
        doc,
        title="Incremental HEAT-MNCH Statistical Analysis Plan",
        subtitle="Superseding the earlier Step A/Step B framing with a current-cycle plus escalation-round roadmap",
        prepared_by="Prepared from the validated repository stack on disk",
        metadata=[
            ("Coverage years", "2020-2024"),
            ("Outcome inventory rows", metrics["total_outcomes"]),
            ("Primary outcomes", metrics["primary"]),
            ("Sensitivity outcomes", metrics["sensitivity"]),
            ("Ready-now endpoints", metrics["ready_now_endpoints"]),
            ("Gold tables in scope", metrics["tables"]),
        ],
        note=(
            "Compatibility note: user-facing plan language is reset in this refresh, "
            "but backend ids and core artifact paths remain unchanged to avoid breaking the dashboard."
        ),
    )

    doc.add_heading("Executive summary", level=1)
    doc.add_paragraph(
        f"The repository is green through Silver RC, Gold Stages 1 to 5, and gold promotion for 2020-2024. "
        f"The available outcome inventory contains {metrics['total_outcomes']} outcome rows across {metrics['groups']} substantive groups "
        f"and {metrics['tables']} gold tables, with {metrics['primary']} primary and {metrics['sensitivity']} sensitivity entries. "
        f"Stage 5 currently certifies {metrics['ready_now_endpoints']} ready-now endpoints and {metrics['dictionary_rows']} dictionary rows."
    )
    doc.add_paragraph(
        "The correct next move is therefore not another data-build sprint. The correct next move is a disciplined analytical escalation plan "
        "that starts with the validated aggregated current-cycle stack and opens an event-level escalation round only when the current stack "
        "cannot answer the question credibly."
    )
    doc.add_paragraph(
        "For this refreshed plan, gold_available_outcomes_latest_round.csv is treated as the authoritative round inventory. "
        "Every CSV-listed variable enters the plan, even if its immediate route is descriptive-only, deferred, matched through an approved equivalence, or blocked pending Step 0 reconciliation."
    )
    doc.add_paragraph(
        "The implemented live roster now also carries three supplementary SIH-grouped child hospitalization rows beyond the original CSV master list: "
        "direct heat hospitalization, indirect heat-sensitive hospitalization, and X30-linked hospitalization. These supplementary rows are screened from Step 1 onward even though they are not separate original CSV entries."
    )
    doc.add_paragraph(
        "The official analytical flow is now steps-only. Operational work should use step-0-freeze through step-6-absolute-burden and steps-run only."
    )
    doc.add_paragraph(
        "Legacy track_a and track_b code remains archived on disk for historical reference, but it is no longer part of the official CLI, the official test suite, or the active execution path."
    )
    doc.add_paragraph(
        "The official internal mart root for the active pipeline is artifacts/analysis/shared_marts. Current and future step outputs should be read from that neutral root rather than from any legacy track-specific artifact directory."
    )
    doc.add_paragraph(
        "This document supersedes the earlier Step A/Step B framing. It keeps compatibility with existing repository and dashboard contracts by "
        "changing planning and presentation language only, while preserving underlying ids and artifact paths."
    )

    doc.add_heading("Current stack aligned to this plan", level=1)
    add_table(doc, ["Layer", "Status", "Current state", "Why it matters"], stage_rows)

    doc.add_heading("Outcome architecture available now", level=1)
    add_table(doc, ["Group", "Primary", "Sensitivity", "Gold table(s)", "Immediate role in the plan"], group_rows)

    doc.add_heading("CSV master inventory rule and live Step 0 coverage", level=1)
    doc.add_paragraph(
        "The plan is governed by the CSV master inventory, not by the current live Step 0 roster alone. "
        "If a variable appears in gold_available_outcomes_latest_round.csv, it must be carried in the plan with an explicit route."
    )
    doc.add_paragraph(
        "Entry into the plan does not force every variable into immediate modelling. Some variables are immediate, some are sensitivity-only, "
        "some stay descriptive-only until Step 1 promotes them, and some are represented through approved equivalence rather than a duplicate live Step 0 row."
    )
    add_table(doc, ["Coverage item", "Count", "Meaning"], reconciliation_summary_rows)

    doc.add_heading("Explicit seasonality contract for the five-year round", level=1)
    doc.add_paragraph(
        "For the 2020-2024 round, seasonality must be explicit in the modelling design. The project should not rely on informal temporal balancing. "
        "Instead, the minimum temporal control set must be chosen by outcome process and then held fixed while threshold and lag choices are evaluated."
    )
    add_table(
        doc,
        ["Outcome family", "Minimum temporal control", "Additional temporal robustness", "Status in this round"],
        SEASONAL_CONTROL_ROWS,
    )

    doc.add_heading("Where seasonality is indispensable", level=1)
    add_table(doc, ["Stage", "Status", "Why it matters"], SEASONAL_STAGE_ROWS)

    doc.add_heading("Incremental roadmap", level=1)
    add_table(doc, ["Step", "Main output", "Advance gate", "If blocked"], roadmap_rows)

    for step in STEP_CONFIG:
        doc.add_heading(f"{step['step']}. {step['title']}", level=2)
        for paragraph in step["paragraphs"]:
            doc.add_paragraph(paragraph)
        doc.add_paragraph("Deliverables")
        add_bullets(doc, step["deliverables"])
        doc.add_paragraph("Decision checklist")
        add_bullets(doc, [f"[ ] {item}" for item in step["gate"]])
        doc.add_paragraph(f"If blocked: {step['if_blocked']}")

    doc.add_heading("Synthesis", level=1)
    add_bullets(
        doc,
        [
            "Promote headline work first for primary perinatal outcomes, maternal canonical outcomes, neonatal or infant mortality, under-five mortality, and the three main child hospitalization outcomes.",
            "Carry the SIH-grouped child direct-heat, indirect-heat, and X30-linked hospitalization totals explicitly in the hospitalization block; allow promotion only when support is adequate, and expect the X30-linked hospitalization outcome to stay sparse unless stronger signal appears.",
            "Run maximum temperature and heat index in parallel from Step 2 onward and choose a headline family only after the Step 3 to Step 4 comparison.",
            "Keep explicit seasonal adjustment mandatory across promoted models in the five-year round; month, season, DOW, conception month, event month, or epidemiologic-week robustness should be matched to the outcome process rather than added inconsistently.",
            "Retain very preterm, preterm with low birth weight, stillbirth >=22 weeks, women aged 15-45 sensitivity denominators, asthma or wheeze hospitalization, and direct or indirect heat any-mention outcomes as sensitivity layers unless they show unusually strong and stable support.",
            "Keep ANC 8+ contacts and the acute-care proxy in the programme, but treat them as supplementary operational indicators rather than headline etiologic results.",
            "Keep the age-split child_u5 mortality rows and all heat-coded child mortality rows in the plan even when they are not yet in the live Step 0 roster; their near-term route is Step 0 expansion plus Step 1 verification, not silent removal.",
            "Treat rare heat-coded child mortality as descriptive or sparse-model surveillance unless event support becomes clearly adequate for stable threshold work.",
            "Use the event-level escalation round only for questions that genuinely need richer gestational timing or specialized sparse-outcome handling.",
        ],
    )

    doc.add_heading("Recommendations", level=1)
    add_table(doc, ["Type", "Action", "Rationale"], RECOMMENDATION_ROWS)

    doc.add_heading("Master decision checklist before the next step", level=1)
    add_bullets(doc, [f"[ ] {item}" for item in MASTER_CHECKLIST])

    doc.add_heading("Safe cleanup applied with compatibility protection", level=1)
    doc.add_paragraph(
        "The cleanup pass deliberately avoided canonical gold tables, manifest directories, statcalc storage, and API ids. "
        "Only superseded plan drafts and obvious disposable clutter were removed or archived."
    )
    if moved_drafts:
        doc.add_paragraph("Archived prior draft plan files")
        add_bullets(doc, moved_drafts)
    if removed_clutter:
        doc.add_paragraph("Removed disposable clutter")
        add_bullets(doc, removed_clutter)

    doc.add_heading("References", level=1)
    add_numbered_references(doc, REFERENCE_LIST)

    doc.add_heading("Appendix A. Technical crosswalk of the current outcome inventory", level=1)
    add_table(doc, ["Group", "Outcome", "Status", "Gold variable", "Gold table"], crosswalk_rows)
    doc.add_heading("Appendix B. CSV master inventory reconciliation against the live Step 0 roster", level=1)
    add_table(
        doc,
        ["Group", "Outcome", "CSV status", "Gold variable", "Gold table", "In live Step 0", "Planned route", "Verification path", "Notes"],
        reconciliation_rows,
    )
    return doc


def main() -> None:
    outcomes = _load_outcome_inventory()
    metrics = _summary_metrics(outcomes)
    _archive_existing_root_doc()
    moved_drafts = _archive_superseded_drafts()
    removed_clutter = _cleanup_hidden_clutter()
    group_rows = _group_summary_rows(outcomes)
    stage_rows = _stage_rows()
    roadmap_rows = _roadmap_rows()
    crosswalk_rows = _crosswalk_rows(outcomes)
    step0_inventory = _load_step0_inventory()
    reconciliation_rows = _inventory_reconciliation_rows(outcomes, step0_inventory)
    reconciliation_summary_rows = _inventory_reconciliation_summary_rows(reconciliation_rows)

    markdown = _build_markdown(
        metrics=metrics,
        moved_drafts=moved_drafts,
        removed_clutter=removed_clutter,
        group_rows=group_rows,
        stage_rows=stage_rows,
        roadmap_rows=roadmap_rows,
        crosswalk_rows=crosswalk_rows,
        reconciliation_summary_rows=reconciliation_summary_rows,
        reconciliation_rows=reconciliation_rows,
    )
    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.write_text(markdown, encoding="utf-8")

    doc = _build_docx(
        metrics=metrics,
        moved_drafts=moved_drafts,
        removed_clutter=removed_clutter,
        group_rows=group_rows,
        stage_rows=stage_rows,
        roadmap_rows=roadmap_rows,
        crosswalk_rows=crosswalk_rows,
        reconciliation_summary_rows=reconciliation_summary_rows,
        reconciliation_rows=reconciliation_rows,
    )
    save_doc(doc, OUTPUT_DOC_PATH)
    shutil.copy2(OUTPUT_DOC_PATH, ROOT_DOC_PATH)

    print(f"Wrote {OUTPUT_DOC_PATH.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_MD_PATH.relative_to(ROOT)}")
    print(f"Refreshed {ROOT_DOC_PATH.name}")


if __name__ == "__main__":
    main()
