from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DAGS_DIR = ROOT / "artifacts" / "analysis" / "step9_rebuild" / "dags"
REGISTRIES_DIR = ROOT / "artifacts" / "analysis" / "step9_rebuild" / "registries"

OUTCOME_ROLE_MAP = REGISTRIES_DIR / "outcome_role_map.csv"
OUTCOME_FAMILY_REGISTRY = REGISTRIES_DIR / "outcome_family_registry.csv"
COVARIATE_FIREWALL = REGISTRIES_DIR / "covariate_firewall.csv"
CLAIM_OUTCOME_MAP = REGISTRIES_DIR / "claim_outcome_map.csv"


STATUS_COLORS = {
    "exposure": "#d55e00",
    "outcome": "#0072b2",
    "estimable_adjustment": "#2c7f4f",
    "stratifier_context": "#f0a202",
    "mediator_do_not_adjust": "#7a7a7a",
    "collider_do_not_adjust": "#ad5d8c",
    "measurement_or_reporting": "#5f9ea0",
    "conceptual_only_not_estimable": "#9c6ade",
}

STATUS_LABELS = {
    "exposure": "Exposure",
    "outcome": "Outcome",
    "estimable_adjustment": "Estimable adjustment",
    "stratifier_context": "Stratifier / context",
    "mediator_do_not_adjust": "Mediator / do not adjust",
    "collider_do_not_adjust": "Collider / do not adjust",
    "measurement_or_reporting": "Measurement or reporting",
    "conceptual_only_not_estimable": "Conceptual only",
}

LEGEND_ORDER = [
    "exposure",
    "outcome",
    "estimable_adjustment",
    "stratifier_context",
    "mediator_do_not_adjust",
    "collider_do_not_adjust",
    "measurement_or_reporting",
    "conceptual_only_not_estimable",
]


OUTCOME_TO_DAG = {
    "perinatal_preterm_birth": "preterm_birth",
    "perinatal_very_preterm_birth": "preterm_birth",
    "perinatal_low_birth_weight": "preterm_birth",
    "perinatal_preterm_low_birth_weight": "preterm_birth",
    "perinatal_gestational_days_lost": "preterm_birth",
    "perinatal_anc_8plus": "preterm_birth",
    "perinatal_stillbirth_gte28": "stillbirth_competing_risks",
    "perinatal_stillbirth_gte22": "stillbirth_competing_risks",
    "child_neonatal_deaths": "neonatal_postnatal_mortality",
    "child_deaths_0_27d": "neonatal_postnatal_mortality",
    "child_infant_deaths": "acute_child_events",
    "child_under5_deaths": "acute_child_events",
    "child_deaths_28_364d": "acute_child_events",
    "child_deaths_1_4y": "acute_child_events",
    "child_total_under5_deaths": "acute_child_events",
    "child_dehydration_admissions": "acute_child_events",
    "child_diarrhea_gi_admissions": "acute_child_events",
    "child_acute_respiratory_admissions": "acute_child_events",
    "child_asthma_wheeze_admissions": "acute_child_events",
    "maternal_canonical_obstetric_admissions": "maternal_operational_admissions",
    "maternal_canonical_hdp": "maternal_operational_admissions",
    "maternal_canonical_preterm_labor": "maternal_operational_admissions",
    "maternal_canonical_gdm": "maternal_operational_admissions",
    "maternal_canonical_aph": "maternal_operational_admissions",
    "maternal_sensitivity_obstetric_admissions": "maternal_operational_admissions",
    "maternal_sensitivity_hdp": "maternal_operational_admissions",
    "maternal_sensitivity_preterm_labor": "maternal_operational_admissions",
    "maternal_sensitivity_gdm": "maternal_operational_admissions",
    "heat_direct_underlying_deaths": "maternal_operational_admissions",
    "heat_indirect_underlying_deaths": "maternal_operational_admissions",
    "heat_x30_any_mention": "maternal_operational_admissions",
    "heat_direct_any_mention": "maternal_operational_admissions",
    "heat_indirect_any_mention": "maternal_operational_admissions",
    "operational_proxy_acute_care": "maternal_operational_admissions",
}


DAG_OUTCOME_ROLE = {
    "perinatal_preterm_birth": "primary_anchor",
    "perinatal_very_preterm_birth": "supportive_corroboration",
    "perinatal_low_birth_weight": "supportive_corroboration",
    "perinatal_preterm_low_birth_weight": "methods_sensitivity",
    "perinatal_gestational_days_lost": "methods_sensitivity",
    "perinatal_anc_8plus": "internal_only_downstream",
    "perinatal_stillbirth_gte28": "key_secondary_anchor",
    "perinatal_stillbirth_gte22": "methods_sensitivity",
    "child_neonatal_deaths": "key_secondary_bridge",
    "child_deaths_0_27d": "audit_only_duplicate",
    "child_infant_deaths": "supportive_corroboration",
    "child_under5_deaths": "supportive_corroboration",
    "child_deaths_28_364d": "supportive_corroboration",
    "child_deaths_1_4y": "supportive_corroboration",
    "child_total_under5_deaths": "audit_only_duplicate",
    "child_dehydration_admissions": "supportive_corroboration",
    "child_diarrhea_gi_admissions": "supportive_corroboration",
    "child_acute_respiratory_admissions": "methods_sensitivity",
    "child_asthma_wheeze_admissions": "methods_sensitivity",
    "maternal_canonical_obstetric_admissions": "methods_sensitivity",
    "maternal_canonical_hdp": "methods_sensitivity",
    "maternal_canonical_preterm_labor": "methods_sensitivity",
    "maternal_canonical_gdm": "methods_sensitivity",
    "maternal_canonical_aph": "methods_sensitivity",
    "maternal_sensitivity_obstetric_admissions": "methods_sensitivity",
    "maternal_sensitivity_hdp": "methods_sensitivity",
    "maternal_sensitivity_preterm_labor": "methods_sensitivity",
    "maternal_sensitivity_gdm": "methods_sensitivity",
    "heat_direct_underlying_deaths": "internal_only_structural",
    "heat_indirect_underlying_deaths": "internal_only_structural",
    "heat_x30_any_mention": "internal_only_structural",
    "heat_direct_any_mention": "internal_only_structural",
    "heat_indirect_any_mention": "internal_only_structural",
    "operational_proxy_acute_care": "internal_only_structural",
}


CLAIM_MAP = {
    "perinatal_preterm_birth": ("C001", "primary", "active"),
    "perinatal_very_preterm_birth": ("C001", "supportive", "active"),
    "perinatal_low_birth_weight": ("C001", "supportive", "active"),
    "perinatal_preterm_low_birth_weight": ("C007", "methods_sensitivity", "active"),
    "perinatal_gestational_days_lost": ("C007", "methods_sensitivity", "active"),
    "perinatal_anc_8plus": ("C010", "internal_only", "active"),
    "perinatal_stillbirth_gte28": ("C002", "key_secondary", "active"),
    "perinatal_stillbirth_gte22": ("C002", "methods_sensitivity", "active"),
    "child_neonatal_deaths": ("C003", "key_secondary", "active"),
    "child_deaths_0_27d": ("C003", "audit_only_removed", "audit_only"),
    "child_infant_deaths": ("C004", "supportive", "active"),
    "child_under5_deaths": ("C004", "supportive", "active"),
    "child_deaths_28_364d": ("C004", "supportive", "active"),
    "child_deaths_1_4y": ("C004", "supportive", "active"),
    "child_total_under5_deaths": ("C004", "audit_only_removed", "audit_only"),
    "child_dehydration_admissions": ("C005", "supportive", "active"),
    "child_diarrhea_gi_admissions": ("C005", "supportive", "active"),
    "child_acute_respiratory_admissions": ("C005", "methods_sensitivity", "active"),
    "child_asthma_wheeze_admissions": ("C005", "methods_sensitivity", "active"),
    "maternal_canonical_obstetric_admissions": ("C008", "methods_sensitivity", "active"),
    "maternal_canonical_hdp": ("C008", "methods_sensitivity", "active"),
    "maternal_canonical_preterm_labor": ("C008", "methods_sensitivity", "active"),
    "maternal_canonical_gdm": ("C008", "methods_sensitivity", "active"),
    "maternal_canonical_aph": ("C008", "methods_sensitivity", "active"),
    "maternal_sensitivity_obstetric_admissions": ("C008", "methods_sensitivity", "active"),
    "maternal_sensitivity_hdp": ("C008", "methods_sensitivity", "active"),
    "maternal_sensitivity_preterm_labor": ("C008", "methods_sensitivity", "active"),
    "maternal_sensitivity_gdm": ("C008", "methods_sensitivity", "active"),
    "heat_direct_underlying_deaths": ("C009", "internal_only", "active"),
    "heat_indirect_underlying_deaths": ("C009", "internal_only", "active"),
    "heat_x30_any_mention": ("C009", "internal_only", "active"),
    "heat_direct_any_mention": ("C009", "internal_only", "active"),
    "heat_indirect_any_mention": ("C009", "internal_only", "active"),
    "operational_proxy_acute_care": ("C010", "internal_only", "active"),
}


DAG_SPECS = [
    {
        "family_id": "acute_child_events",
        "title": "Acute Child Events DAG",
        "filename_stem": "dag_family_acute_child_events",
        "primary_estimand_id": "E001",
        "claim_scope": "Supportive acute child mortality and pediatric admissions within municipalities",
        "authorship_status": "project_authored_dag",
        "authorship_note": "Project-authored DAG informed by climate-health time-series methods and child heat studies because no strong journal-published DAG matched the exact acute within-municipality Brazilian MNCH design.",
        "scope_statement": "This DAG represents short-term heat exposure analyses for acute child mortality and hospitalization outcomes in municipality-day panels.",
        "purpose": "To freeze the acute causal architecture for supportive child mortality and morbidity analyses without collapsing supportive and sensitivity respiratory outcomes into the same claim tier.",
        "primary_estimable_adjustment_set": [
            "municipality fixed effects",
            "calendar time controls",
            "day-of-week indicators",
            "relative humidity when the primary exposure family is maximum daily temperature",
        ],
        "conditional_guidance": [
            "IDH is retained as a stratifier or effect-modifier, not as the primary within-municipality identification strategy.",
            "GINI remains robustness-only and should not replace the primary IDH contextual axis.",
        ],
        "prohibited_adjustments": [
            "acute-care utilization proxies or downstream care-seeking descendants",
            "simultaneous inclusion of competing exposure families in the same primary model",
        ],
        "narrative_paragraphs": [
            "The acute child DAG preserves within-municipality identification as the core causal architecture. Stable municipal context is handled by fixed effects, while time-varying seasonal structure is controlled through explicit calendar design rather than through ad hoc covariate accumulation.",
            "Respiratory outcomes remain in the DAG because they share the same short-term heat-exposure architecture, but their claim status is frozen as methods sensitivity rather than core corroboration.",
            "Acute-care proxy fields are retained structurally to show why they are not valid total-effect adjustment variables.",
        ],
        "nodes": [
            {"id": "calendar_time", "label": "Calendar time\nand seasonality", "role": "time_control", "status": "estimable_adjustment", "note": "Year, seasonal, and day-of-week structure." , "position": [0.10, 0.82]},
            {"id": "municipality_fe", "label": "Municipality\nfixed effects", "role": "baseline_context", "status": "estimable_adjustment", "note": "Primary within-municipality identification layer.", "position": [0.10, 0.46]},
            {"id": "humidity", "label": "Relative humidity\n(conditional)", "role": "meteorology", "status": "estimable_adjustment", "note": "Used only in temperature-based primary models.", "position": [0.34, 0.82]},
            {"id": "heat_exposure", "label": "Short-term heat\nexposure", "role": "exposure", "status": "exposure", "note": "TMAX-centered acute exposure family.", "position": [0.38, 0.56]},
            {"id": "idh_context", "label": "Municipal IDH\ncontext", "role": "context", "status": "stratifier_context", "note": "Effect-modification and heterogeneity axis.", "position": [0.32, 0.22]},
            {"id": "service_pressure", "label": "Health-service\npressure", "role": "system_context", "status": "conceptual_only_not_estimable", "note": "Conceptual short-term vulnerability node.", "position": [0.63, 0.82]},
            {"id": "reporting_practice", "label": "Reporting and\ncoding practice", "role": "measurement", "status": "measurement_or_reporting", "note": "Measurement and ascertainment quality.", "position": [0.64, 0.28]},
            {"id": "acute_outcome", "label": "Acute child events", "role": "outcome", "status": "outcome", "note": "Mortality, dehydration, diarrhea/GI, and respiratory admissions.", "position": [0.86, 0.56]},
            {"id": "acute_care_proxy", "label": "Acute-care proxy\n(do not adjust)", "role": "proxy", "status": "collider_do_not_adjust", "note": "Internal-only descendant not for total-effect adjustment.", "position": [0.88, 0.16]},
        ],
        "edges": [
            {"source": "calendar_time", "target": "heat_exposure", "narrative": "Seasonality shapes daily heat exposure."},
            {"source": "calendar_time", "target": "acute_outcome", "narrative": "Seasonality also shapes short-term child-event risk."},
            {"source": "municipality_fe", "target": "heat_exposure", "narrative": "Stable municipal structure affects ambient heat distribution."},
            {"source": "municipality_fe", "target": "acute_outcome", "narrative": "Stable municipal structure affects baseline child-event risk."},
            {"source": "humidity", "target": "heat_exposure", "narrative": "Humidity modifies apparent thermal burden in temperature-led models."},
            {"source": "idh_context", "target": "acute_outcome", "narrative": "Municipal development modifies vulnerability and care access."},
            {"source": "service_pressure", "target": "acute_outcome", "narrative": "Health-service pressure affects vulnerability and observed admissions."},
            {"source": "reporting_practice", "target": "acute_outcome", "narrative": "Coding and reporting practice affect measured event counts."},
            {"source": "heat_exposure", "target": "acute_outcome", "narrative": "Primary acute heat-health pathway of interest."},
            {"source": "heat_exposure", "target": "acute_care_proxy", "narrative": "Heat may influence downstream care-seeking and acute proxy counts."},
            {"source": "acute_outcome", "target": "acute_care_proxy", "narrative": "Outcome severity also drives downstream acute-care proxies."},
        ],
        "analog_rows": [
            {
                "citation_key": "brimicombe2024under5africaheat",
                "source_tier": "published_conceptual_framework_high_priority",
                "source_quality_note": "Multi-country pooled child heat study aligned with acute mortality architecture.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained the acute daily exposure and short-lag child-event framing.",
                "added_nodes": "municipality fixed effects; acute-care proxy; IDH context",
                "rejected_nodes": "none",
                "adaptation_rationale": "Study informs outcome domain and acute time scale but does not supply a published DAG tailored to Brazilian municipality-day identification.",
            },
            {
                "citation_key": "basu2015temperatureinfantmortality",
                "source_tier": "published_journal_dag_high_priority",
                "source_quality_note": "Paediatric and Perinatal Epidemiology temperature-infant mortality design analogue.",
                "used_as": "partial_inspiration",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained acute temperature, lag, and age-specific child mortality logic.",
                "added_nodes": "humidity; municipality fixed effects; acute-care proxy",
                "rejected_nodes": "air-pollution co-exposure branch as a default Round 2 node",
                "adaptation_rationale": "The study is methodologically aligned but not presented as a published DAG and does not include the Step 9 covariate firewall.",
            },
            {
                "citation_key": "wang2021laggedclimatehealth",
                "source_tier": "systematic_review_or_methods_framework",
                "source_quality_note": "High-quality lag-structure methods review.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Supports explicit lag declaration and acute timing discipline.",
                "added_nodes": "none",
                "rejected_nodes": "none",
                "adaptation_rationale": "Used to justify explicit lag architecture rather than node structure.",
            },
        ],
    },
    {
        "family_id": "preterm_birth",
        "title": "Preterm Birth and Related Birth Phenotypes DAG",
        "filename_stem": "dag_family_preterm_birth",
        "primary_estimand_id": "E002",
        "claim_scope": "Flagship gestational timing and supportive live-birth phenotypes",
        "authorship_status": "literature_adapted_dag",
        "authorship_note": "Literature-adapted DAG drawing on published perinatal DAG methods, live-birth bias work, and pregnancy heat studies, then translated to the Step 9 heat-exposure estimand and Brazilian source systems.",
        "scope_statement": "This DAG represents gestational heat exposure and its relationship to delivery before 37 completed weeks and related live-birth phenotypes.",
        "purpose": "To freeze the flagship gestational DAG and make explicit why prenatal care intensity and delivery interventions stay outside total-effect adjustment sets.",
        "primary_estimable_adjustment_set": [
            "maternal age",
            "schooling",
            "parity",
            "plurality",
            "fetal or newborn sex",
            "baseline pregnancy timing",
        ],
        "conditional_guidance": [
            "Maternal race or skin color is included when available and documented.",
            "Congenital anomaly remains restriction or sensitivity only rather than routine default adjustment.",
        ],
        "prohibited_adjustments": [
            "ANC 8+ or similar prenatal-care intensity descendants",
            "delivery interventions triggered after heat exposure and before birth ascertainment",
        ],
        "narrative_paragraphs": [
            "The preterm DAG is the flagship causal structure for Step 9. It preserves maternal and fetal baseline confounders while explicitly fencing care-process intensity and delivery interventions out of the total-effect adjustment set.",
            "Low birth weight and very preterm birth are retained inside the same DAG because they sit on the same pregnancy-time architecture, but their claim status remains supportive rather than primary.",
            "The DAG is literature-adapted rather than copied because the project estimand is heat-specific, Brazilian, and intended for individual-level gestational modeling after the `SINASC` rebuild.",
        ],
        "nodes": [
            {"id": "pregnancy_profile", "label": "Maternal age,\nschooling, parity,\nplurality", "role": "baseline_profile", "status": "estimable_adjustment", "note": "Required core confounders after the individual rebuild.", "position": [0.10, 0.78]},
            {"id": "pregnancy_timing", "label": "Conception timing\nand calendar season", "role": "timing", "status": "estimable_adjustment", "note": "Baseline timing structure for gestational exposure history.", "position": [0.10, 0.42]},
            {"id": "idh_context", "label": "Municipal IDH\ncontext", "role": "context", "status": "stratifier_context", "note": "Contextual inequality axis for later interaction work.", "position": [0.32, 0.18]},
            {"id": "heat_exposure", "label": "Gestational heat\nexposure history", "role": "exposure", "status": "exposure", "note": "Pregnancy-scale TMAX-centered exposure.", "position": [0.38, 0.56]},
            {"id": "fetal_characteristics", "label": "Fetal sex and\nbaseline fetal profile", "role": "fetal_profile", "status": "estimable_adjustment", "note": "Required fetal-level confounder or precision covariate.", "position": [0.33, 0.82]},
            {"id": "congenital_anomaly", "label": "Congenital anomaly\n(policy only)", "role": "restriction", "status": "conceptual_only_not_estimable", "note": "Restriction or sensitivity policy rather than default adjustment.", "position": [0.60, 0.82]},
            {"id": "anc_intensity", "label": "ANC 8+ contacts\n(do not adjust)", "role": "care_process", "status": "mediator_do_not_adjust", "note": "Downstream care-process node.", "position": [0.64, 0.26]},
            {"id": "delivery_intervention", "label": "Delivery intervention\n(do not adjust)", "role": "clinical_action", "status": "mediator_do_not_adjust", "note": "Post-exposure clinical response node.", "position": [0.69, 0.58]},
            {"id": "preterm_outcome", "label": "Preterm birth and\nrelated phenotypes", "role": "outcome", "status": "outcome", "note": "Preterm birth, very preterm birth, low birth weight, and composite sensitivity.", "position": [0.88, 0.56]},
        ],
        "edges": [
            {"source": "pregnancy_profile", "target": "heat_exposure", "narrative": "Baseline social and reproductive factors shape heat exposure context."},
            {"source": "pregnancy_profile", "target": "preterm_outcome", "narrative": "Baseline profile affects preterm and related birth-phenotype risk."},
            {"source": "pregnancy_timing", "target": "heat_exposure", "narrative": "Conception timing determines gestational heat windows."},
            {"source": "pregnancy_timing", "target": "preterm_outcome", "narrative": "Baseline timing also affects delivery timing risk."},
            {"source": "idh_context", "target": "heat_exposure", "narrative": "Municipal development shapes environmental and structural exposure context."},
            {"source": "idh_context", "target": "preterm_outcome", "narrative": "Municipal development also shapes vulnerability and care environment."},
            {"source": "fetal_characteristics", "target": "preterm_outcome", "narrative": "Fetal characteristics affect delivery timing and phenotype expression."},
            {"source": "congenital_anomaly", "target": "preterm_outcome", "narrative": "Congenital anomaly may affect birth timing but is handled through policy rather than routine adjustment."},
            {"source": "heat_exposure", "target": "anc_intensity", "narrative": "Heat may alter prenatal-care attendance or service access."},
            {"source": "heat_exposure", "target": "delivery_intervention", "narrative": "Heat-related pregnancy stress may trigger clinical action."},
            {"source": "heat_exposure", "target": "preterm_outcome", "narrative": "Primary gestational heat effect of interest."},
            {"source": "anc_intensity", "target": "preterm_outcome", "narrative": "Care intensity can affect observed preterm risk but is downstream for total-effect estimation."},
            {"source": "delivery_intervention", "target": "preterm_outcome", "narrative": "Delivery intervention lies on the pathway from heat-triggered pregnancy stress to early delivery."},
        ],
        "analog_rows": [
            {
                "citation_key": "bandoli2016perinataldags",
                "source_tier": "published_journal_dag_high_priority",
                "source_quality_note": "Direct perinatal DAG methods paper in Paediatric and Perinatal Epidemiology.",
                "used_as": "direct_template",
                "direct_dag_available": "yes",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained the principle of explicit confounder screening in common perinatal outcomes.",
                "added_nodes": "gestational heat exposure; ANC 8+; municipal IDH context",
                "rejected_nodes": "drug-specific exposure branches",
                "adaptation_rationale": "The paper provides a directly relevant perinatal DAG workflow but not a heat-specific pregnancy DAG.",
            },
            {
                "citation_key": "hough2023earlydeliverytemperature",
                "source_tier": "published_conceptual_framework_high_priority",
                "source_quality_note": "Strong survival-oriented pregnancy heat comparator.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained pregnancy timing and event-time orientation.",
                "added_nodes": "care-process mediator firewall",
                "rejected_nodes": "study-specific exposure parameterization",
                "adaptation_rationale": "Used to align the DAG with the flagship gestational hazard estimand.",
            },
            {
                "citation_key": "goin2021livebirthbias",
                "source_tier": "published_journal_dag_high_priority",
                "source_quality_note": "Live-birth bias paper with explicit DAGs for pregnancy selection bias.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "yes",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained the need to avoid silent conditioning structures in pregnancy outcomes.",
                "added_nodes": "delivery intervention and ANC firewall nodes",
                "rejected_nodes": "spontaneous abortion selection node as a default visible node in the flagship DAG",
                "adaptation_rationale": "Included to justify why live-birth conditioning and downstream care processes must remain explicit.",
            },
        ],
    },
    {
        "family_id": "stillbirth_competing_risks",
        "title": "Stillbirth With Competing Risks DAG",
        "filename_stem": "dag_family_stillbirth_competing_risks",
        "primary_estimand_id": "E003",
        "claim_scope": "Key secondary stillbirth hazard among fetuses at risk",
        "authorship_status": "project_authored_dag",
        "authorship_note": "Project-authored DAG anchored in published stillbirth, live-birth bias, and competing-mediator literature because no direct heat-stillbirth DAG matched the desired fetuses-at-risk estimand.",
        "scope_statement": "This DAG represents gestational heat exposure, fetal risk-set logic, and competing live-birth pathways for stillbirth outcomes.",
        "purpose": "To freeze the stillbirth causal architecture and make explicit that live birth is a competing event structure rather than a routine adjustment covariate.",
        "primary_estimable_adjustment_set": [
            "maternal age",
            "schooling",
            "parity",
            "plurality",
            "fetal sex when available",
            "baseline pregnancy timing",
        ],
        "conditional_guidance": [
            "Maternal race or skin color is included when available.",
            "Congenital anomaly remains restriction or sensitivity only.",
        ],
        "prohibited_adjustments": [
            "ANC 8+ or other downstream care-process variables",
            "delivery interventions triggered after heat exposure",
            "conditioning on live birth as if it were a pre-exposure covariate",
        ],
        "narrative_paragraphs": [
            "The stillbirth DAG differs from the flagship preterm DAG because the estimand is defined among fetuses at risk and the live-birth path is a competing event, not a standard adjustment variable.",
            "This structure is project-authored because the exact Brazilian heat-stillbirth question is not matched by a journal-published DAG, but the design is anchored in published stillbirth heat evidence and explicit live-birth-bias literature.",
            "The `>=22 weeks` threshold is kept inside the same DAG as a sensitivity outcome, not as a second headline estimand.",
        ],
        "nodes": [
            {"id": "pregnancy_profile", "label": "Maternal age,\nschooling, parity,\nplurality", "role": "baseline_profile", "status": "estimable_adjustment", "note": "Core gestational confounders.", "position": [0.10, 0.78]},
            {"id": "pregnancy_timing", "label": "Conception timing\nand calendar season", "role": "timing", "status": "estimable_adjustment", "note": "Determines gestational heat windows.", "position": [0.10, 0.42]},
            {"id": "heat_exposure", "label": "Gestational heat\nexposure history", "role": "exposure", "status": "exposure", "note": "Pregnancy-scale heat exposure.", "position": [0.37, 0.56]},
            {"id": "fetal_risk_set", "label": "Ongoing pregnancy\nrisk set", "role": "risk_set", "status": "estimable_adjustment", "note": "Defines fetuses-at-risk logic over gestation.", "position": [0.36, 0.84]},
            {"id": "idh_context", "label": "Municipal IDH\ncontext", "role": "context", "status": "stratifier_context", "note": "Contextual effect-modification axis.", "position": [0.33, 0.20]},
            {"id": "delivery_intervention", "label": "Delivery intervention\n(do not adjust)", "role": "clinical_action", "status": "mediator_do_not_adjust", "note": "Post-exposure clinical actions.", "position": [0.62, 0.30]},
            {"id": "live_birth_competing", "label": "Live birth pathway\n(do not adjust)", "role": "competing_event", "status": "collider_do_not_adjust", "note": "Competing event structure not treated as a baseline covariate.", "position": [0.66, 0.78]},
            {"id": "stillbirth_outcome", "label": "Stillbirth outcome", "role": "outcome", "status": "outcome", "note": "Primary >=28-week and sensitivity >=22-week outcome thresholds.", "position": [0.88, 0.56]},
            {"id": "reporting_practice", "label": "Fetal-death coding\nand reporting", "role": "measurement", "status": "measurement_or_reporting", "note": "Measurement and source-quality node.", "position": [0.64, 0.12]},
        ],
        "edges": [
            {"source": "pregnancy_profile", "target": "heat_exposure", "narrative": "Baseline maternal profile shapes exposure context."},
            {"source": "pregnancy_profile", "target": "stillbirth_outcome", "narrative": "Baseline maternal profile shapes stillbirth risk."},
            {"source": "pregnancy_timing", "target": "heat_exposure", "narrative": "Pregnancy timing shapes gestational heat windows."},
            {"source": "pregnancy_timing", "target": "fetal_risk_set", "narrative": "Timing defines entry into and duration of the fetal risk set."},
            {"source": "fetal_risk_set", "target": "live_birth_competing", "narrative": "Ongoing pregnancy can culminate in live birth as a competing event."},
            {"source": "fetal_risk_set", "target": "stillbirth_outcome", "narrative": "Ongoing pregnancy can culminate in stillbirth."},
            {"source": "idh_context", "target": "heat_exposure", "narrative": "Context shapes structural exposure conditions."},
            {"source": "idh_context", "target": "stillbirth_outcome", "narrative": "Context also shapes vulnerability and care environments."},
            {"source": "heat_exposure", "target": "delivery_intervention", "narrative": "Heat-related compromise may trigger clinical action."},
            {"source": "heat_exposure", "target": "live_birth_competing", "narrative": "Heat may shift the pathway toward early live birth."},
            {"source": "heat_exposure", "target": "stillbirth_outcome", "narrative": "Primary stillbirth pathway of interest."},
            {"source": "delivery_intervention", "target": "live_birth_competing", "narrative": "Clinical action may divert the pregnancy toward live birth."},
            {"source": "reporting_practice", "target": "stillbirth_outcome", "narrative": "Fetal-death coding and investigation quality affect measurement."},
        ],
        "analog_rows": [
            {
                "citation_key": "auger2016stillbirthheat",
                "source_tier": "published_conceptual_framework_high_priority",
                "source_quality_note": "High-impact stillbirth and heat paper in International Journal of Epidemiology.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained stillbirth as a dedicated pregnancy heat endpoint.",
                "added_nodes": "fetuses-at-risk structure; live birth competing event; Brazilian source-quality node",
                "rejected_nodes": "none",
                "adaptation_rationale": "The paper supports the endpoint and timing question but does not provide a published DAG for the current estimand.",
            },
            {
                "citation_key": "goin2021livebirthbias",
                "source_tier": "published_journal_dag_high_priority",
                "source_quality_note": "Published pregnancy-selection DAGs directly relevant to live-birth conditioning.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "yes",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained explicit live-birth selection concerns and vulnerability structure.",
                "added_nodes": "competing live-birth event rather than simple live-birth restriction",
                "rejected_nodes": "spontaneous abortion as the visible selected node",
                "adaptation_rationale": "Used to keep selection bias explicit while translating the DAG to stillbirth among fetuses at risk.",
            },
            {
                "citation_key": "ananth2025competingmediators",
                "source_tier": "published_journal_dag_high_priority",
                "source_quality_note": "Published DAG with competing mediators and perinatal mortality pathways.",
                "used_as": "partial_inspiration",
                "direct_dag_available": "yes",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained competing-pathway logic between obstetric action and adverse perinatal outcomes.",
                "added_nodes": "fetal risk-set and contextual inequality nodes",
                "rejected_nodes": "abruption-specific exposure structure",
                "adaptation_rationale": "Helpful for competing pathways but not directly transferable as a heat-stillbirth DAG.",
            },
        ],
    },
    {
        "family_id": "neonatal_postnatal_mortality",
        "title": "Neonatal Postnatal Mortality DAG",
        "filename_stem": "dag_family_neonatal_postnatal_mortality",
        "primary_estimand_id": "E004",
        "claim_scope": "Key secondary neonatal bridge estimand by day of life",
        "authorship_status": "project_authored_dag",
        "authorship_note": "Project-authored DAG using published infant heat studies and pregnancy survival work as analogues because no strong journal DAG matched day-of-life neonatal mortality under the Step 9 bridge design.",
        "scope_statement": "This DAG represents postnatal daily heat exposure and neonatal death risk across day-of-life risk sets.",
        "purpose": "To freeze the neonatal bridge DAG and keep day of life separate from the non-neonatal acute child DAG.",
        "primary_estimable_adjustment_set": [
            "municipality fixed effects",
            "calendar time controls",
            "day of life risk-set indicators",
            "relative humidity when primary exposure is TMAX",
        ],
        "conditional_guidance": [
            "Birth characteristics are retained as stratifiers or sensitivity descriptors, not default total-effect adjusters in Round 2.",
            "If linkage improves in later rounds, the DAG can support individual survival analysis without reopening the core firewall.",
        ],
        "prohibited_adjustments": [
            "acute-care proxy descendants",
            "post-exposure clinical deterioration markers",
        ],
        "narrative_paragraphs": [
            "The neonatal DAG is separated from the broader acute child DAG because the time scale is day of life, not generic calendar day alone.",
            "Birth characteristics remain visible because neonatal vulnerability is strongly shaped by gestational age, congenital anomaly, and birthweight, but Round 2 freezes them as stratifier or sensitivity variables rather than automatic default adjusters for the total effect of heat.",
            "The `Deaths 0-27 days` slice remains audit-only because neonatal deaths already cover the intended bridge estimand.",
        ],
        "nodes": [
            {"id": "calendar_time", "label": "Calendar time\nand seasonality", "role": "time_control", "status": "estimable_adjustment", "note": "Calendar-day structure for postnatal heat exposure.", "position": [0.10, 0.82]},
            {"id": "municipality_fe", "label": "Municipality\nfixed effects", "role": "baseline_context", "status": "estimable_adjustment", "note": "Primary location-level identification layer.", "position": [0.10, 0.44]},
            {"id": "day_of_life", "label": "Day of life\nrisk set", "role": "time_scale", "status": "estimable_adjustment", "note": "Risk-set time scale, not a confounder.", "position": [0.32, 0.82]},
            {"id": "heat_exposure", "label": "Postnatal heat\nexposure", "role": "exposure", "status": "exposure", "note": "Daily postnatal heat family.", "position": [0.40, 0.56]},
            {"id": "birth_profile", "label": "Gestational age,\nbirthweight, sex,\nanomaly", "role": "birth_profile", "status": "stratifier_context", "note": "Stratifier or sensitivity branch in Round 2.", "position": [0.34, 0.18]},
            {"id": "idh_context", "label": "Municipal IDH\ncontext", "role": "context", "status": "stratifier_context", "note": "Contextual heterogeneity axis.", "position": [0.56, 0.82]},
            {"id": "reporting_practice", "label": "Mortality coding\nand linkage quality", "role": "measurement", "status": "measurement_or_reporting", "note": "Measurement and linkage-quality node.", "position": [0.66, 0.24]},
            {"id": "neonatal_outcome", "label": "Neonatal death", "role": "outcome", "status": "outcome", "note": "Day-of-life bridge outcome.", "position": [0.86, 0.56]},
            {"id": "acute_care_proxy", "label": "Acute-care proxy\n(do not adjust)", "role": "proxy", "status": "collider_do_not_adjust", "note": "Internal-only descendant.", "position": [0.88, 0.14]},
        ],
        "edges": [
            {"source": "calendar_time", "target": "heat_exposure", "narrative": "Calendar structure shapes daily postnatal heat."},
            {"source": "calendar_time", "target": "neonatal_outcome", "narrative": "Calendar structure also shapes baseline neonatal mortality."},
            {"source": "municipality_fe", "target": "heat_exposure", "narrative": "Stable municipal structure shapes exposure context."},
            {"source": "municipality_fe", "target": "neonatal_outcome", "narrative": "Stable municipal structure shapes baseline neonatal mortality risk."},
            {"source": "day_of_life", "target": "neonatal_outcome", "narrative": "Risk changes sharply across day of life."},
            {"source": "birth_profile", "target": "neonatal_outcome", "narrative": "Birth profile shapes neonatal vulnerability."},
            {"source": "idh_context", "target": "neonatal_outcome", "narrative": "Municipal development shapes structural neonatal vulnerability and care context."},
            {"source": "heat_exposure", "target": "neonatal_outcome", "narrative": "Primary postnatal heat effect of interest."},
            {"source": "heat_exposure", "target": "acute_care_proxy", "narrative": "Heat influences downstream care use and proxy counts."},
            {"source": "neonatal_outcome", "target": "acute_care_proxy", "narrative": "Evolving neonatal illness or death also shapes proxy counts."},
            {"source": "reporting_practice", "target": "neonatal_outcome", "narrative": "Mortality coding and linkage quality affect measured outcome counts."},
        ],
        "analog_rows": [
            {
                "citation_key": "basu2015temperatureinfantmortality",
                "source_tier": "published_conceptual_framework_high_priority",
                "source_quality_note": "Paediatric and Perinatal Epidemiology infant heat study with neonatal and post-neonatal differentiation.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained day-specific neonatal and infant heat timing logic.",
                "added_nodes": "day-of-life risk set; linkage-quality node; audit-only duplicate rule",
                "rejected_nodes": "air pollution co-exposure branch as a default visible node",
                "adaptation_rationale": "The study provides the most relevant neonatal heat analogue but not a published DAG aligned with the Step 9 bridge estimand.",
            },
            {
                "citation_key": "hough2023earlydeliverytemperature",
                "source_tier": "published_conceptual_framework_high_priority",
                "source_quality_note": "Pregnancy survival framework supports time-scale discipline.",
                "used_as": "partial_inspiration",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained event-time discipline and separation of timing scale from exposure family.",
                "added_nodes": "day-of-life bridge architecture",
                "rejected_nodes": "pregnancy-only exposure history as the primary scale",
                "adaptation_rationale": "Used to preserve explicit time-scale logic while moving to postnatal neonatal risk sets.",
            },
        ],
    },
    {
        "family_id": "maternal_operational_admissions",
        "title": "Maternal Operational Admissions and Internal Sentinels DAG",
        "filename_stem": "dag_family_maternal_operational_admissions",
        "primary_estimand_id": "E001",
        "claim_scope": "Secondary maternal operational outcomes and internal-only rare thermal sentinels",
        "authorship_status": "literature_adapted_dag",
        "authorship_note": "Literature-adapted DAG using published evidence-synthesis DAG methods in perinatal epidemiology plus Brazilian SIH/SUS validation evidence, then translated to the administrative maternal-outcome context.",
        "scope_statement": "This DAG represents acute heat exposure, administrative obstetric admissions, denominator sensitivities, and internal-only thermal sentinel coding in municipality-day panels.",
        "purpose": "To keep maternal operational outcomes clearly secondary while documenting why rare heat-coded sentinels and acute-care proxies stay structural and internal-only.",
        "primary_estimable_adjustment_set": [
            "municipality fixed effects",
            "calendar time controls",
            "day-of-week indicators when operationally relevant",
        ],
        "conditional_guidance": [
            "IDH remains the primary contextual modifier.",
            "GINI remains robustness-only.",
        ],
        "prohibited_adjustments": [
            "ANC 8+ or other downstream care-process descendants",
            "acute-care proxy fields",
        ],
        "narrative_paragraphs": [
            "The maternal operational DAG is deliberately administrative and system-facing. It allows acute maternal outcomes to be analysed without promoting them into flagship etiologic claims.",
            "Rare heat-coded causes remain visible only as internal structural sentinels because the coding pathway is sparse and vulnerable to measurement instability.",
            "The DAG is literature-adapted for process rather than copied for content: evidence-synthesis DAG methods support the way the DAG is built, while Brazilian SIH/SUS validation anchors the source-system meaning.",
        ],
        "nodes": [
            {"id": "calendar_time", "label": "Calendar time\nand seasonality", "role": "time_control", "status": "estimable_adjustment", "note": "Operational event timing structure.", "position": [0.10, 0.82]},
            {"id": "municipality_fe", "label": "Municipality\nfixed effects", "role": "baseline_context", "status": "estimable_adjustment", "note": "Location-specific baseline risk and structure.", "position": [0.10, 0.44]},
            {"id": "heat_exposure", "label": "Short-term heat\nexposure", "role": "exposure", "status": "exposure", "note": "Acute maternal exposure family.", "position": [0.37, 0.56]},
            {"id": "idh_context", "label": "Municipal IDH\ncontext", "role": "context", "status": "stratifier_context", "note": "Primary contextual heterogeneity axis.", "position": [0.32, 0.18]},
            {"id": "system_pressure", "label": "Health-system\npressure", "role": "system_context", "status": "conceptual_only_not_estimable", "note": "Conceptual short-term service-pressure node.", "position": [0.58, 0.82]},
            {"id": "coding_practice", "label": "SIH/SUS coding\nand ascertainment", "role": "measurement", "status": "measurement_or_reporting", "note": "Administrative coding and episode-definition quality.", "position": [0.60, 0.24]},
            {"id": "anc_intensity", "label": "ANC 8+ contacts\n(do not adjust)", "role": "care_process", "status": "mediator_do_not_adjust", "note": "Care-process descendant excluded from total-effect adjustment.", "position": [0.64, 0.42]},
            {"id": "maternal_outcome", "label": "Maternal operational\noutcomes", "role": "outcome", "status": "outcome", "note": "Obstetric admissions and maternal operational phenotypes.", "position": [0.86, 0.56]},
            {"id": "internal_proxy", "label": "Rare heat coding /\nacute-care proxy", "role": "internal_proxy", "status": "collider_do_not_adjust", "note": "Internal-only structural branch.", "position": [0.87, 0.14]},
        ],
        "edges": [
            {"source": "calendar_time", "target": "heat_exposure", "narrative": "Calendar structure shapes daily heat exposure."},
            {"source": "calendar_time", "target": "maternal_outcome", "narrative": "Calendar structure also shapes observed maternal operational counts."},
            {"source": "municipality_fe", "target": "heat_exposure", "narrative": "Stable municipal structure shapes acute exposure context."},
            {"source": "municipality_fe", "target": "maternal_outcome", "narrative": "Stable municipal structure shapes baseline administrative outcome risk."},
            {"source": "idh_context", "target": "heat_exposure", "narrative": "Municipal development shapes structural exposure conditions."},
            {"source": "idh_context", "target": "maternal_outcome", "narrative": "Municipal development shapes access and vulnerability."},
            {"source": "system_pressure", "target": "maternal_outcome", "narrative": "System pressure shapes observed admissions and coding thresholds."},
            {"source": "coding_practice", "target": "maternal_outcome", "narrative": "Coding and ascertainment affect measured administrative outcomes."},
            {"source": "heat_exposure", "target": "anc_intensity", "narrative": "Heat may alter care-seeking and prenatal-care continuity."},
            {"source": "heat_exposure", "target": "maternal_outcome", "narrative": "Primary acute heat effect of interest for the secondary maternal domain."},
            {"source": "heat_exposure", "target": "internal_proxy", "narrative": "Heat can alter rare thermal coding or downstream acute proxy counts."},
            {"source": "maternal_outcome", "target": "internal_proxy", "narrative": "Observed administrative outcomes can also influence internal proxy fields."},
        ],
        "analog_rows": [
            {
                "citation_key": "hebert2025thirdstageescdag",
                "source_tier": "published_journal_dag_high_priority",
                "source_quality_note": "Published evidence-synthesis DAG workflow in Paediatric and Perinatal Epidemiology.",
                "used_as": "direct_template",
                "direct_dag_available": "yes",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained evidence-synthesis DAG construction discipline for perinatal administrative questions.",
                "added_nodes": "acute heat exposure; Brazilian administrative coding branch",
                "rejected_nodes": "third-stage management exposure specifics",
                "adaptation_rationale": "The article directly supports how the DAG should be constructed even though the exposure and outcome differ.",
            },
            {
                "citation_key": "domingues_obstetric_hospitalization_validation_2024",
                "source_tier": "systematic_review_or_methods_framework",
                "source_quality_note": "Brazilian validation support for obstetric hospitalization use in SIH/SUS.",
                "used_as": "conceptual_anchor",
                "direct_dag_available": "no",
                "adaptation_required": "yes",
                "project_authored_allowed": "yes",
                "borrowed_structure_summary": "Retained administrative validity boundaries for maternal hospitalization outcomes.",
                "added_nodes": "rare thermal coding and acute-care proxy branch",
                "rejected_nodes": "none",
                "adaptation_rationale": "Used to define what the administrative system can and cannot support, not as a published DAG.",
            },
        ],
    },
]


FIREWALL_ROWS = [
    {
        "dag_family_id": "acute_child_events",
        "estimand_id": "E001",
        "covariate_name": "municipality_fixed_effect",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "all active acute child mortality and hospitalization outcomes",
        "source_system": "project_design",
        "official_source_citation_key": "",
        "methods_support_citation_key": "wang2021laggedclimatehealth",
        "brazil_validation_citation_key": "silveira_under5_mortality_brazil_heat_2025",
        "rationale": "Primary within-municipality identification layer.",
        "notes": "Design-level control, not source-native field semantics.",
    },
    {
        "dag_family_id": "acute_child_events",
        "estimand_id": "E001",
        "covariate_name": "calendar_time_controls",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "all active acute child mortality and hospitalization outcomes",
        "source_system": "project_design",
        "official_source_citation_key": "",
        "methods_support_citation_key": "wang2021laggedclimatehealth",
        "brazil_validation_citation_key": "",
        "rationale": "Time-varying confounding control for acute models.",
        "notes": "Includes seasonal structure and year controls.",
    },
    {
        "dag_family_id": "acute_child_events",
        "estimand_id": "E001",
        "covariate_name": "day_of_week",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "all active acute child mortality and hospitalization outcomes",
        "source_system": "project_design",
        "official_source_citation_key": "",
        "methods_support_citation_key": "wang2021laggedclimatehealth",
        "brazil_validation_citation_key": "",
        "rationale": "Short-term temporal control.",
        "notes": "",
    },
    {
        "dag_family_id": "acute_child_events",
        "estimand_id": "E001",
        "covariate_name": "relative_humidity",
        "covariate_bucket": "adjust",
        "policy_strength": "conditional",
        "applies_to_scope": "temperature-led acute models only",
        "source_system": "climate_source_family",
        "official_source_citation_key": "",
        "methods_support_citation_key": "wang2021laggedclimatehealth",
        "brazil_validation_citation_key": "",
        "rationale": "Meteorological control only when exposure family is TMAX.",
        "notes": "Do not enter together with heat index as a competing primary exposure family.",
    },
    {
        "dag_family_id": "acute_child_events",
        "estimand_id": "E005",
        "covariate_name": "IDH",
        "covariate_bucket": "stratifier",
        "policy_strength": "required",
        "applies_to_scope": "heterogeneity and effect-modification layer",
        "source_system": "context_registry",
        "official_source_citation_key": "",
        "methods_support_citation_key": "brimicombe2024under5africaheat",
        "brazil_validation_citation_key": "silveira_under5_mortality_brazil_heat_2025",
        "rationale": "Primary contextual effect-modification axis.",
        "notes": "Not a replacement for within-municipality identification.",
    },
    {
        "dag_family_id": "acute_child_events",
        "estimand_id": "E005",
        "covariate_name": "GINI",
        "covariate_bucket": "stratifier",
        "policy_strength": "conditional",
        "applies_to_scope": "robustness-only heterogeneity repeats",
        "source_system": "context_registry",
        "official_source_citation_key": "",
        "methods_support_citation_key": "brimicombe2024under5africaheat",
        "brazil_validation_citation_key": "",
        "rationale": "Robustness contextual inequality axis.",
        "notes": "Not co-primary with IDH.",
    },
    {
        "dag_family_id": "acute_child_events",
        "estimand_id": "E001",
        "covariate_name": "acute_care_proxy",
        "covariate_bucket": "do_not_adjust",
        "policy_strength": "forbidden",
        "applies_to_scope": "all acute child outcomes",
        "source_system": "project_operational_proxy",
        "official_source_citation_key": "",
        "methods_support_citation_key": "basu2015temperatureinfantmortality",
        "brazil_validation_citation_key": "",
        "rationale": "Downstream descendant and potential collider.",
        "notes": "Internal-only or descriptive use only.",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "maternal_age",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "flagship and supportive live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "bandoli2016perinataldags",
        "brazil_validation_citation_key": "szwarcwald_sinasc_evaluation_2019",
        "rationale": "Core individual confounder.",
        "notes": "Requires individual rebuild in Round 3A.",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "schooling",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "flagship and supportive live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "bandoli2016perinataldags",
        "brazil_validation_citation_key": "szwarcwald_sinasc_evaluation_2019",
        "rationale": "Core socioeconomic confounder.",
        "notes": "",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "parity",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "flagship and supportive live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "bandoli2016perinataldags",
        "brazil_validation_citation_key": "szwarcwald_sinasc_evaluation_2019",
        "rationale": "Core reproductive-history confounder.",
        "notes": "",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "plurality",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "flagship and supportive live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "bandoli2016perinataldags",
        "brazil_validation_citation_key": "szwarcwald_sinasc_evaluation_2019",
        "rationale": "Strong confounder of preterm and birthweight phenotypes.",
        "notes": "",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "fetal_or_newborn_sex",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "flagship and supportive live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "bandoli2016perinataldags",
        "brazil_validation_citation_key": "",
        "rationale": "Biologic confounder or precision covariate.",
        "notes": "",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E005",
        "covariate_name": "maternal_race_or_skin_color",
        "covariate_bucket": "adjust",
        "policy_strength": "if_available",
        "applies_to_scope": "flagship and supportive live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "szwarcwald_sinasc_evaluation_2019",
        "rationale": "Equity-relevant confounder when available.",
        "notes": "",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "congenital_anomaly",
        "covariate_bucket": "do_not_adjust",
        "policy_strength": "restriction_or_sensitivity_only",
        "applies_to_scope": "flagship and supportive live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "",
        "rationale": "Requires explicit restriction or sensitivity policy rather than routine default adjustment.",
        "notes": "",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "ANC_8PLUS",
        "covariate_bucket": "mediator",
        "policy_strength": "forbidden",
        "applies_to_scope": "all live-birth phenotypes",
        "source_system": "SINASC",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "bandoli2016perinataldags",
        "brazil_validation_citation_key": "",
        "rationale": "Downstream care-process mediator.",
        "notes": "May be used only for descriptive or mediation-specific work.",
    },
    {
        "dag_family_id": "preterm_birth",
        "estimand_id": "E002",
        "covariate_name": "delivery_interventions",
        "covariate_bucket": "mediator",
        "policy_strength": "forbidden",
        "applies_to_scope": "all live-birth phenotypes",
        "source_system": "project_operational_or_linked_records",
        "official_source_citation_key": "",
        "methods_support_citation_key": "bandoli2016perinataldags",
        "brazil_validation_citation_key": "",
        "rationale": "Post-exposure clinical actions on the causal pathway.",
        "notes": "",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "maternal_age",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "primary and sensitivity stillbirth thresholds",
        "source_system": "SIM_or_linked_fetal_records",
        "official_source_citation_key": "brasil_ms_sim_do_dictionary_2025",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "barbeiro_fetal_deaths_brazil_review_2015",
        "rationale": "Core stillbirth confounder.",
        "notes": "May be complemented by linked `SINASC` or fetal-death data in later rounds.",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "schooling",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "primary and sensitivity stillbirth thresholds",
        "source_system": "SIM_or_linked_fetal_records",
        "official_source_citation_key": "brasil_ms_sim_do_dictionary_2025",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "barros_fetal_mortality_brazil_2019",
        "rationale": "Core socioeconomic confounder.",
        "notes": "",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "parity",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "primary and sensitivity stillbirth thresholds",
        "source_system": "SIM_or_linked_fetal_records",
        "official_source_citation_key": "brasil_ms_sim_do_dictionary_2025",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "",
        "rationale": "Core reproductive-history confounder.",
        "notes": "",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "plurality",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "primary and sensitivity stillbirth thresholds",
        "source_system": "SIM_or_linked_fetal_records",
        "official_source_citation_key": "brasil_ms_sim_do_dictionary_2025",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "",
        "rationale": "Strong fetal-risk confounder.",
        "notes": "",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "fetal_sex",
        "covariate_bucket": "adjust",
        "policy_strength": "if_available",
        "applies_to_scope": "primary and sensitivity stillbirth thresholds",
        "source_system": "SIM_or_linked_fetal_records",
        "official_source_citation_key": "brasil_ms_sim_do_dictionary_2025",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "",
        "rationale": "Relevant fetal-risk factor when available.",
        "notes": "",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "congenital_anomaly",
        "covariate_bucket": "do_not_adjust",
        "policy_strength": "restriction_or_sensitivity_only",
        "applies_to_scope": "primary and sensitivity stillbirth thresholds",
        "source_system": "SIM_or_linked_fetal_records",
        "official_source_citation_key": "brasil_ms_sim_do_dictionary_2025",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "",
        "rationale": "Restriction or sensitivity policy only.",
        "notes": "",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "ANC_8PLUS",
        "covariate_bucket": "mediator",
        "policy_strength": "forbidden",
        "applies_to_scope": "stillbirth thresholds and fetal-risk estimands",
        "source_system": "SINASC_or_linked_records",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "goin2021livebirthbias",
        "brazil_validation_citation_key": "",
        "rationale": "Downstream care-process mediator.",
        "notes": "",
    },
    {
        "dag_family_id": "stillbirth_competing_risks",
        "estimand_id": "E003",
        "covariate_name": "live_birth_competing_event",
        "covariate_bucket": "collider",
        "policy_strength": "forbidden",
        "applies_to_scope": "stillbirth hazard among fetuses at risk",
        "source_system": "project_risk_set_logic",
        "official_source_citation_key": "",
        "methods_support_citation_key": "ananth2025competingmediators",
        "brazil_validation_citation_key": "",
        "rationale": "Competing-event structure rather than baseline covariate.",
        "notes": "",
    },
    {
        "dag_family_id": "neonatal_postnatal_mortality",
        "estimand_id": "E004",
        "covariate_name": "municipality_fixed_effect",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "neonatal bridge outcome",
        "source_system": "project_design",
        "official_source_citation_key": "",
        "methods_support_citation_key": "basu2015temperatureinfantmortality",
        "brazil_validation_citation_key": "silveira_under5_mortality_brazil_heat_2025",
        "rationale": "Location-specific identification layer.",
        "notes": "",
    },
    {
        "dag_family_id": "neonatal_postnatal_mortality",
        "estimand_id": "E004",
        "covariate_name": "calendar_time_controls",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "neonatal bridge outcome",
        "source_system": "project_design",
        "official_source_citation_key": "",
        "methods_support_citation_key": "basu2015temperatureinfantmortality",
        "brazil_validation_citation_key": "",
        "rationale": "Time-varying confounding control for neonatal daily analysis.",
        "notes": "",
    },
    {
        "dag_family_id": "neonatal_postnatal_mortality",
        "estimand_id": "E004",
        "covariate_name": "day_of_life",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "neonatal bridge outcome",
        "source_system": "project_risk_set_logic",
        "official_source_citation_key": "",
        "methods_support_citation_key": "basu2015temperatureinfantmortality",
        "brazil_validation_citation_key": "",
        "rationale": "Time scale for neonatal hazard.",
        "notes": "Risk-set scale, not a confounder.",
    },
    {
        "dag_family_id": "neonatal_postnatal_mortality",
        "estimand_id": "E004",
        "covariate_name": "birth_characteristics",
        "covariate_bucket": "stratifier",
        "policy_strength": "conditional",
        "applies_to_scope": "neonatal bridge outcome",
        "source_system": "SINASC_or_linked_records",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "hough2023earlydeliverytemperature",
        "brazil_validation_citation_key": "henriques_sinasc_accuracy_2019",
        "rationale": "Stratifier or sensitivity descriptor rather than default total-effect adjustment.",
        "notes": "Includes gestational age, birthweight, sex, and anomaly when linkage permits.",
    },
    {
        "dag_family_id": "neonatal_postnatal_mortality",
        "estimand_id": "E005",
        "covariate_name": "IDH",
        "covariate_bucket": "stratifier",
        "policy_strength": "required",
        "applies_to_scope": "heterogeneity layer",
        "source_system": "context_registry",
        "official_source_citation_key": "",
        "methods_support_citation_key": "basu2015temperatureinfantmortality",
        "brazil_validation_citation_key": "",
        "rationale": "Primary contextual modifier.",
        "notes": "",
    },
    {
        "dag_family_id": "neonatal_postnatal_mortality",
        "estimand_id": "E004",
        "covariate_name": "acute_care_proxy",
        "covariate_bucket": "do_not_adjust",
        "policy_strength": "forbidden",
        "applies_to_scope": "neonatal bridge outcome",
        "source_system": "project_operational_proxy",
        "official_source_citation_key": "",
        "methods_support_citation_key": "basu2015temperatureinfantmortality",
        "brazil_validation_citation_key": "",
        "rationale": "Downstream descendant and potential collider.",
        "notes": "",
    },
    {
        "dag_family_id": "maternal_operational_admissions",
        "estimand_id": "E001",
        "covariate_name": "municipality_fixed_effect",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "maternal operational outcomes",
        "source_system": "project_design",
        "official_source_citation_key": "",
        "methods_support_citation_key": "hebert2025thirdstageescdag",
        "brazil_validation_citation_key": "domingues_obstetric_hospitalization_validation_2024",
        "rationale": "Primary location-specific identification layer.",
        "notes": "",
    },
    {
        "dag_family_id": "maternal_operational_admissions",
        "estimand_id": "E001",
        "covariate_name": "calendar_time_controls",
        "covariate_bucket": "adjust",
        "policy_strength": "required",
        "applies_to_scope": "maternal operational outcomes",
        "source_system": "project_design",
        "official_source_citation_key": "",
        "methods_support_citation_key": "hebert2025thirdstageescdag",
        "brazil_validation_citation_key": "",
        "rationale": "Time-varying confounding control.",
        "notes": "",
    },
    {
        "dag_family_id": "maternal_operational_admissions",
        "estimand_id": "E005",
        "covariate_name": "IDH",
        "covariate_bucket": "stratifier",
        "policy_strength": "required",
        "applies_to_scope": "heterogeneity layer for maternal operational outcomes",
        "source_system": "context_registry",
        "official_source_citation_key": "",
        "methods_support_citation_key": "domingues_obstetric_hospitalization_validation_2024",
        "brazil_validation_citation_key": "domingues_obstetric_hospitalization_validation_2024",
        "rationale": "Primary contextual modifier.",
        "notes": "",
    },
    {
        "dag_family_id": "maternal_operational_admissions",
        "estimand_id": "E005",
        "covariate_name": "GINI",
        "covariate_bucket": "stratifier",
        "policy_strength": "conditional",
        "applies_to_scope": "robustness-only heterogeneity layer",
        "source_system": "context_registry",
        "official_source_citation_key": "",
        "methods_support_citation_key": "domingues_obstetric_hospitalization_validation_2024",
        "brazil_validation_citation_key": "",
        "rationale": "Robustness-only contextual axis.",
        "notes": "",
    },
    {
        "dag_family_id": "maternal_operational_admissions",
        "estimand_id": "E001",
        "covariate_name": "ANC_8PLUS",
        "covariate_bucket": "mediator",
        "policy_strength": "forbidden",
        "applies_to_scope": "maternal operational outcomes",
        "source_system": "SINASC_or_project_linkage",
        "official_source_citation_key": "brasil_ms_sinasc_dn_dictionary_2012",
        "methods_support_citation_key": "hebert2025thirdstageescdag",
        "brazil_validation_citation_key": "",
        "rationale": "Downstream care-process node.",
        "notes": "",
    },
    {
        "dag_family_id": "maternal_operational_admissions",
        "estimand_id": "E001",
        "covariate_name": "acute_care_proxy",
        "covariate_bucket": "do_not_adjust",
        "policy_strength": "forbidden",
        "applies_to_scope": "maternal operational outcomes and internal sentinels",
        "source_system": "project_operational_proxy",
        "official_source_citation_key": "",
        "methods_support_citation_key": "domingues_obstetric_hospitalization_validation_2024",
        "brazil_validation_citation_key": "",
        "rationale": "Internal-only downstream proxy branch.",
        "notes": "",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wrap(text: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def render_dag_png(spec: dict[str, object], png_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(spec["title"], fontsize=18, fontweight="bold", pad=20)

    node_lookup = {node["id"]: node for node in spec["nodes"]}
    box_width = 0.18
    box_height = 0.10

    for edge in spec["edges"]:
        source = node_lookup[edge["source"]]
        target = node_lookup[edge["target"]]
        axis.annotate(
            "",
            xy=target["position"],
            xytext=source["position"],
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#444444"},
            zorder=1,
        )

    for node in spec["nodes"]:
        x, y = node["position"]
        color = STATUS_COLORS[node["status"]]
        rect = patches.FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor=color,
            alpha=0.90,
            zorder=2,
        )
        axis.add_patch(rect)
        axis.text(
            x,
            y,
            node["label"],
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            zorder=3,
        )

    legend_handles = [
        patches.Patch(color=STATUS_COLORS[key], label=STATUS_LABELS[key]) for key in LEGEND_ORDER
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_dag_markdown(
    spec: dict[str, object],
    covered_outcomes: list[dict[str, str]],
    analog_rows: list[dict[str, str]],
) -> str:
    lines: list[str] = [
        f"# {spec['title']}",
        "",
        "## Purpose",
        "",
        str(spec["purpose"]),
        "",
        "## Scope Statement",
        "",
        str(spec["scope_statement"]),
        "",
        "## Literature Preference and Authorship Status",
        "",
        f"- Authorship status: `{spec['authorship_status']}`",
        f"- Authorship note: {spec['authorship_note']}",
        "",
        "## Narrative Rationale",
        "",
    ]
    for paragraph in spec["narrative_paragraphs"]:
        lines.extend([paragraph, ""])
    lines.extend(
        [
            "## Primary Estimable Adjustment Set",
            "",
        ]
    )
    for item in spec["primary_estimable_adjustment_set"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Conditional Guidance", ""])
    for item in spec["conditional_guidance"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Prohibited Adjustments", ""])
    for item in spec["prohibited_adjustments"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Node Register", "", "| Node ID | Label | Role | Status | Note |", "| --- | --- | --- | --- | --- |"])
    for node in spec["nodes"]:
        label = node["label"].replace("\n", "<br>")
        status_label = STATUS_LABELS[node["status"]]
        lines.append(f"| `{node['id']}` | {label} | {node['role']} | {status_label} | {node['note']} |")
    lines.extend(["", "## Edge Register", "", "| Source | Target | Narrative |", "| --- | --- | --- |"])
    for edge in spec["edges"]:
        lines.append(f"| `{edge['source']}` | `{edge['target']}` | {edge['narrative']} |")
    lines.extend(["", "## Covered Round 2 Outcomes", "", "| Inventory row | Public label | Analysis status | DAG role | Claim tier |", "| --- | --- | --- | --- | --- |"])
    for outcome in covered_outcomes:
        lines.append(
            f"| `{outcome['inventory_row_id']}` | {outcome['public_label']} | {outcome['analysis_universe_status']} | {outcome['dag_outcome_role']} | {outcome['claim_tier']} |"
        )
    lines.extend(["", "## Analogue Literature Trail", "", "| Citation key | Source tier | Used as | Direct DAG available | Adaptation rationale |", "| --- | --- | --- | --- | --- |"])
    for row in analog_rows:
        lines.append(
            f"| `{row['citation_key']}` | {row['source_tier']} | {row['used_as']} | {row['direct_dag_available']} | {row['adaptation_rationale']} |"
        )
    lines.append("")
    return "\n".join(lines)


def load_outcome_maps() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    role_rows = read_csv(OUTCOME_ROLE_MAP)
    family_rows = {
        row["inventory_row_id"]: row for row in read_csv(OUTCOME_FAMILY_REGISTRY)
    }
    return role_rows, family_rows


def build_package() -> None:
    DAGS_DIR.mkdir(parents=True, exist_ok=True)
    role_rows, family_rows = load_outcome_maps()

    dag_rows: list[dict[str, str]] = []
    dag_outcome_rows: list[dict[str, str]] = []
    analog_rows: list[dict[str, str]] = []
    claim_rows: list[dict[str, str]] = []

    role_rows_by_id = {row["inventory_row_id"]: row for row in role_rows}

    for outcome_id, dag_family_id in OUTCOME_TO_DAG.items():
        role_row = role_rows_by_id[outcome_id]
        claim_id, claim_tier, claim_status = CLAIM_MAP[outcome_id]
        dag_outcome_rows.append(
            {
                "inventory_row_id": outcome_id,
                "public_label": role_row["public_label"],
                "analysis_universe_status": role_row["analysis_universe_status"],
                "dag_family_id": dag_family_id,
                "dag_outcome_role": DAG_OUTCOME_ROLE[outcome_id],
                "manuscript_visibility": role_row["manuscript_visibility"],
                "claim_tier": claim_tier,
                "notes": role_row["notes"],
            }
        )
        claim_rows.append(
            {
                "inventory_row_id": outcome_id,
                "analytic_subfamily_id": role_row["analytic_subfamily_id"],
                "dag_family_id": dag_family_id,
                "claim_id": claim_id,
                "claim_tier": claim_tier,
                "claim_status_round2": claim_status,
                "headline_eligible": "yes" if claim_tier in {"primary", "key_secondary"} else "no",
                "manuscript_visibility": role_row["manuscript_visibility"],
                "notes": role_row["notes"],
            }
        )

    dag_outcomes_by_family: dict[str, list[dict[str, str]]] = {}
    for row in dag_outcome_rows:
        dag_outcomes_by_family.setdefault(row["dag_family_id"], []).append(row)

    for spec in DAG_SPECS:
        family_id = spec["family_id"]
        filename_stem = spec["filename_stem"]
        json_path = DAGS_DIR / f"{filename_stem}.json"
        md_path = DAGS_DIR / f"{filename_stem}.md"
        png_path = DAGS_DIR / f"{filename_stem}.png"

        json_payload = {
            "family_id": family_id,
            "title": spec["title"],
            "purpose": spec["purpose"],
            "scope_statement": spec["scope_statement"],
            "primary_estimand_id": spec["primary_estimand_id"],
            "claim_scope": spec["claim_scope"],
            "authorship_status": spec["authorship_status"],
            "authorship_note": spec["authorship_note"],
            "primary_estimable_adjustment_set": spec["primary_estimable_adjustment_set"],
            "conditional_guidance": spec["conditional_guidance"],
            "prohibited_adjustments": spec["prohibited_adjustments"],
            "narrative_paragraphs": spec["narrative_paragraphs"],
            "nodes": spec["nodes"],
            "edges": spec["edges"],
            "analog_rows": spec["analog_rows"],
        }
        json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        render_dag_png(spec, png_path)

        covered_outcomes = sorted(
            dag_outcomes_by_family[family_id],
            key=lambda row: (row["analysis_universe_status"], row["public_label"]),
        )
        analog_rows_family: list[dict[str, str]] = []
        for row in spec["analog_rows"]:
            analog_rows_family.append(
                {
                    "dag_family_id": family_id,
                    "citation_key": row["citation_key"],
                    "source_tier": row["source_tier"],
                    "source_quality_note": row["source_quality_note"],
                    "used_as": row["used_as"],
                    "direct_dag_available": row["direct_dag_available"],
                    "adaptation_required": row["adaptation_required"],
                    "project_authored_allowed": row["project_authored_allowed"],
                    "borrowed_structure_summary": row["borrowed_structure_summary"],
                    "added_nodes": row["added_nodes"],
                    "rejected_nodes": row["rejected_nodes"],
                    "adaptation_rationale": row["adaptation_rationale"],
                }
            )
        analog_rows.extend(analog_rows_family)
        md_path.write_text(build_dag_markdown(spec, covered_outcomes, analog_rows_family), encoding="utf-8")

        dag_rows.append(
            {
                "family_id": family_id,
                "title": spec["title"],
                "json_path": str(json_path.relative_to(ROOT)),
                "markdown_path": str(md_path.relative_to(ROOT)),
                "png_path": str(png_path.relative_to(ROOT)),
                "node_count": str(len(spec["nodes"])),
                "edge_count": str(len(spec["edges"])),
                "covered_outcome_count": str(len(covered_outcomes)),
                "authorship_status": spec["authorship_status"],
                "primary_estimand_id": spec["primary_estimand_id"],
                "claim_scope": spec["claim_scope"],
            }
        )

    write_csv(
        DAGS_DIR / "dag_inventory.csv",
        dag_rows,
        [
            "family_id",
            "title",
            "json_path",
            "markdown_path",
            "png_path",
            "node_count",
            "edge_count",
            "covered_outcome_count",
            "authorship_status",
            "primary_estimand_id",
            "claim_scope",
        ],
    )

    write_csv(
        DAGS_DIR / "dag_outcome_map.csv",
        sorted(dag_outcome_rows, key=lambda row: row["inventory_row_id"]),
        [
            "inventory_row_id",
            "public_label",
            "analysis_universe_status",
            "dag_family_id",
            "dag_outcome_role",
            "manuscript_visibility",
            "claim_tier",
            "notes",
        ],
    )

    write_csv(
        DAGS_DIR / "dag_analog_literature_map.csv",
        analog_rows,
        [
            "dag_family_id",
            "citation_key",
            "source_tier",
            "source_quality_note",
            "used_as",
            "direct_dag_available",
            "adaptation_required",
            "project_authored_allowed",
            "borrowed_structure_summary",
            "added_nodes",
            "rejected_nodes",
            "adaptation_rationale",
        ],
    )

    adjustment_lines = ["# Round 2 Adjustment Sets", ""]
    for spec in DAG_SPECS:
        adjustment_lines.extend([f"## {spec['title']}", ""])
        covered = [row["public_label"] for row in dag_outcomes_by_family[spec["family_id"]]]
        adjustment_lines.append("Covered outcomes: " + ", ".join(f"`{label}`" for label in sorted(covered)))
        adjustment_lines.extend(["", "Primary estimable set:"])
        for item in spec["primary_estimable_adjustment_set"]:
            adjustment_lines.append(f"- {item}")
        adjustment_lines.extend(["", "Conditional guidance:"])
        for item in spec["conditional_guidance"]:
            adjustment_lines.append(f"- {item}")
        adjustment_lines.extend(["", "Prohibited descendants or forbidden adjustments:"])
        for item in spec["prohibited_adjustments"]:
            adjustment_lines.append(f"- {item}")
        adjustment_lines.extend(["", f"Interpretive note: {spec['authorship_note']}", ""])
    (DAGS_DIR / "adjustment_sets.md").write_text("\n".join(adjustment_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Round 2 DAG Package Report",
        "",
        "- Round ID: `2`",
        f"- DAG family count: {len(DAG_SPECS)}",
        f"- Figure count: {len(DAG_SPECS)}",
        "",
        "## Family Coverage",
        "",
    ]
    for row in dag_rows:
        family_outcomes = dag_outcomes_by_family[row["family_id"]]
        summary = {}
        for item in family_outcomes:
            summary[item["claim_tier"]] = summary.get(item["claim_tier"], 0) + 1
        report_lines.append(
            f"- `{row['family_id']}`: {row['covered_outcome_count']} covered outcomes; claim summary = {summary}; authorship = `{row['authorship_status']}`"
        )
    report_lines.extend(
        [
            "",
            "## Package Notes",
            "",
            "- Project-authored DAGs are allowed when the literature does not provide a strong directly transferable journal-published DAG.",
            "- Audit-only duplicates remain in the source inventory but are excluded from modeling, figures, and claims.",
            "- Acute respiratory outcomes remain inside the acute child DAG while staying frozen as methods sensitivity outcomes.",
            "- Rare thermal sentinel outcomes remain structural and internal-only.",
            "",
        ]
    )
    (DAGS_DIR / "dag_package_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    claim_rows_sorted = sorted(claim_rows, key=lambda row: row["inventory_row_id"])
    write_csv(
        CLAIM_OUTCOME_MAP,
        claim_rows_sorted,
        [
            "inventory_row_id",
            "analytic_subfamily_id",
            "dag_family_id",
            "claim_id",
            "claim_tier",
            "claim_status_round2",
            "headline_eligible",
            "manuscript_visibility",
            "notes",
        ],
    )

    write_csv(
        COVARIATE_FIREWALL,
        FIREWALL_ROWS,
        [
            "dag_family_id",
            "estimand_id",
            "covariate_name",
            "covariate_bucket",
            "policy_strength",
            "applies_to_scope",
            "source_system",
            "official_source_citation_key",
            "methods_support_citation_key",
            "brazil_validation_citation_key",
            "rationale",
            "notes",
        ],
    )


if __name__ == "__main__":
    build_package()
