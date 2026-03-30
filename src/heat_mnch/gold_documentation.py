from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


CANONICAL_GOLD_TABLES = (
    "gold_mnch_perinatal",
    "gold_mnch_mortality",
    "gold_mnch_panel",
)
DERIVED_GOLD_VIEWS = (
    "gold_analysis_child_u5",
    "gold_analysis_wra_15_49",
    "gold_analysis_wra_15_45_sens",
)
DOCUMENTED_GOLD_ITEMS = (*CANONICAL_GOLD_TABLES, *DERIVED_GOLD_VIEWS)

TABLE_LAYER = {
    "gold_mnch_perinatal": "gold",
    "gold_mnch_mortality": "gold",
    "gold_mnch_panel": "gold",
    "gold_analysis_child_u5": "derived_gold",
    "gold_analysis_wra_15_49": "derived_gold",
    "gold_analysis_wra_15_45_sens": "derived_gold",
}

TABLE_AGE_SCOPE = {
    "gold_mnch_perinatal": "all births / no age truncation",
    "gold_mnch_mortality": "children under-5",
    "gold_mnch_panel": "mixed municipality-day comparative mart",
    "gold_analysis_child_u5": "0-59 months / 0-4 completed years",
    "gold_analysis_wra_15_49": "women 15-49 canonical",
    "gold_analysis_wra_15_45_sens": "women 15-45 sensitivity",
}

TABLE_SOURCE_PROVENANCE = {
    "gold_mnch_perinatal": (
        "Canonical gold built from approved silver_sinasc_event, silver_stillbirth_daily, "
        "silver_stillbirth_individual, refined silver_sih_daily, and climate_canonical_daily."
    ),
    "gold_mnch_mortality": (
        "Canonical gold built from approved silver_child_u5_sim_daily, silver_sim_individual, "
        "silver_sinasc_daily, silver_pop_under5, climate_canonical_daily, and the active ICD-10 dictionary "
        "for approved cause-group outputs."
    ),
    "gold_mnch_panel": (
        "Canonical harmonized municipality-day mart built from approved gold_mnch_perinatal, "
        "gold_mnch_mortality, and silver_pop_wra_15_49 without redefining endpoint semantics."
    ),
    "gold_analysis_child_u5": "Derived analytical view built only from approved gold_mnch_mortality.",
    "gold_analysis_wra_15_49": "Derived analytical view built from approved gold_mnch_perinatal and silver_pop_wra_15_49.",
    "gold_analysis_wra_15_45_sens": (
        "Sensitivity-only derived analytical view built from approved gold_mnch_perinatal and "
        "silver_pop_wra_15_45_sensitivity."
    ),
}

TABLE_SCHEMA_FILES = {
    "gold_mnch_perinatal": "gold_mnch_perinatal_diario_municipio.schema.csv",
    "gold_mnch_mortality": "gold_mnch_mortality_diario_municipio.schema.csv",
    "gold_mnch_panel": "gold_mnch_panel_diario_municipio.schema.csv",
}

CLIMATE_COLUMNS = ("TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "FONTE_CLIMA_FINAL")

LEGACY_COMPATIBILITY_FIELDS = {
    "STILLBIRTH_TOTAL",
    "NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS",
    "OFFSET_LOG_NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS",
    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS",
    "NEONATAL_DEATHS_TOTAL",
    "INFANT_DEATHS_TOTAL",
    "UNDER5_DEATHS_TOTAL",
    "LIVE_BIRTHS_TOTAL",
    "UNDER5_POPULATION",
    "CRUDE_RATE_NEONATAL",
    "CRUDE_RATE_U1",
    "CRUDE_RATE_U5",
}

PRIMARY_OUTCOME_FIELDS = {
    "PRETERM_FLAG",
    "LOW_BIRTH_WEIGHT_LT2500_TOTAL",
    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
    "STILLBIRTH_PRIMARY_GTE28_FLAG",
    "STILLBIRTH_TOTAL_PRIMARY_GTE28",
    "DEATHS_NEONATAL",
    "DEATHS_U1",
    "DEATHS_U5",
    "PRETERM_BIRTHS_CANONICAL_TOTAL",
    "DEATHS_0_27D",
    "DEATHS_28_364D",
    "DEATHS_1_4Y",
    "DEATHS_UNDER5_TOTAL",
    "UNDERLYING_BASIC_CAUSE_CHILD_HEAT_THERMAL_PRIMARY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL",
    "UNDERLYING_BASIC_CAUSE_SLOW_MOVING_HEAT_ANY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
}

SECONDARY_OR_SENSITIVITY_FIELDS = {
    "VERY_PRETERM_FLAG",
    "GEST_DAYS_LOST_BELOW_37W",
    "STILLBIRTH_SENS_GTE22_FLAG",
    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
    "DIABETES_PREGNANCY_BROAD_TOTAL",
    "HEATWAVE_LEN_P95",
    "HEAT_INDEX_APPROX_C",
    "ANY_MENTION_CHILD_FLUID_ELECTROLYTE_SENSITIVITY_TOTAL",
    "ANY_MENTION_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
    "ANY_MENTION_CHILD_RENAL_ACUTE_SENSITIVITY_TOTAL",
    "ANY_MENTION_CHILD_MALNUTRITION_SLOW_MOVING_SENSITIVITY_TOTAL",
    "ANY_MENTION_DIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL",
    "ANY_MENTION_SLOW_MOVING_HEAT_ANY_TOTAL",
}

MATERNAL_PHENOTYPE_FIELDS = {
    "OBSTETRIC_ADMISSIONS_TOTAL",
    "OBSTETRIC_URGENT_TOTAL",
    "OBSTETRIC_ICU_TOTAL",
    "HDP_ANY_TOTAL",
    "HDP_SEVERE_TOTAL",
    "HDP_GESTATIONAL_HTN_TOTAL",
    "HDP_PREECLAMPSIA_TOTAL",
    "HDP_ECLAMPSIA_TOTAL",
    "PRETERM_LABOR_TOTAL",
    "PRETERM_LABOR_URGENT_TOTAL",
    "PRETERM_LABOR_ICU_TOTAL",
    "PRETERM_LABOR_DEATH_TOTAL",
    "APH_ANTEPARTUM_TOTAL",
    "APH_ANTEPARTUM_URGENT_TOTAL",
    "APH_ANTEPARTUM_ICU_TOTAL",
    "APH_ANTEPARTUM_DEATH_TOTAL",
    "GDM_GESTATIONAL_TOTAL",
    "GDM_GESTATIONAL_URGENT_TOTAL",
    "GDM_GESTATIONAL_ICU_TOTAL",
    "GDM_GESTATIONAL_DEATH_TOTAL",
}

NOT_APPLICABLE_DENOMINATOR = "not_applicable_count_or_descriptor"

PLAIN_LABEL_OVERRIDES = {
    "GEST_AGE_CANONICAL": "Canonical gestational age",
    "PRETERM_FLAG": "Preterm births below 37 completed weeks",
    "VERY_PRETERM_FLAG": "Very preterm births below 32 completed weeks",
    "GEST_DAYS_LOST_BELOW_37W": "Gestational days lost below 37 weeks",
    "HEAT_INDEX_APPROX_C": "Approximate heat index in Celsius",
    "DENOMINATOR_ZERO_FLAG": "Denominator equals zero flag",
    "DENOMINATOR_MISSING_FLAG": "Denominator missing flag",
    "RATE_NOT_COMPUTABLE_FLAG": "Rate not computable flag",
    "OFFSET_VALID_FLAG": "Offset valid flag",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": "Climate upstream station quarantine flag",
    "CLIMATE_RECORD_ANOMALY_FLAG": "Climate record anomaly flag",
    "CLIMATE_CORRECTION_PATH": "Climate correction path label",
    "SOURCE_DERIVED_VIEW": "Derived view lineage marker",
    "VIEW_ROLE_NOTE": "Derived view role note",
    "WRA_AGE_SCOPE_NOTE": "Women of reproductive age scope note",
    "CHILD_VIEW_AGE_LOGIC_NOTE": "Child age logic note",
    "SENSITIVITY_ONLY_FLAG": "Sensitivity-only view flag",
}

DEFINITION_OVERRIDES = {
    "GEST_AGE_CANONICAL": (
        "Daily mean canonical gestational age aggregated from the approved event-level SINASC pathway. "
        "The repository-standard canonical scale is completed weeks."
    ),
    "PRETERM_FLAG": "Daily count of live births with canonical gestational age below 37 completed weeks.",
    "VERY_PRETERM_FLAG": "Daily count of live births with canonical gestational age below 32 completed weeks; strict subset of PRETERM_FLAG.",
    "GEST_DAYS_LOST_BELOW_37W": (
        "Daily sum of max(0, 37*7 minus canonical gestational age in days used for derivation); "
        "kept as a secondary sensitivity outcome and never as the primary preterm endpoint."
    ),
    "HEAT_INDEX_APPROX_C": (
        "Approximate composite heat-index exposure derived from the final merged climate series. "
        "This is a sensitivity exposure and may inherit mixed-source lineage."
    ),
    "DENOMINATOR_ZERO_FLAG": (
        "Safety flag set to 1 when the required denominator for the relevant descriptive rate equals zero."
    ),
    "DENOMINATOR_MISSING_FLAG": (
        "Safety flag set to 1 when the required denominator for the relevant descriptive rate is missing."
    ),
    "RATE_NOT_COMPUTABLE_FLAG": (
        "Safety flag set to 1 when the descriptive rate must remain NA because the denominator is missing or non-positive."
    ),
    "OFFSET_VALID_FLAG": (
        "Safety flag set to 1 only when the required denominator is positive and the corresponding model offset is valid."
    ),
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": (
        "Indicator equal to 1 when the municipality-day climate lineage was protected from an invalid upstream INMET station "
        "and the standard source-selection logic fell back to ERA5."
    ),
    "CLIMATE_RECORD_ANOMALY_FLAG": (
        "Indicator equal to 1 when the municipality-day preliminary climate tuple was nullified because the selected record "
        "failed the hard-bound or cross-source anomaly screen."
    ),
    "CLIMATE_CORRECTION_PATH": (
        "Compact provenance label recording whether the climate row required no correction, upstream station quarantine to ERA5, "
        "neighbor-based anomaly imputation, or fallback imputation after anomaly detection."
    ),
    "SOURCE_DERIVED_VIEW": "Explicit lineage marker showing that the record was emitted by a Stage 4 analytical derived view.",
    "VIEW_ROLE_NOTE": "Documentation note stating that the analytical view does not replace canonical gold tables.",
    "WRA_AGE_SCOPE_NOTE": "Documentation note distinguishing the canonical 15-49 view from the 15-45 sensitivity view.",
    "CHILD_VIEW_AGE_LOGIC_NOTE": "Documentation note stating the explicit child age bands used in the under-5 analytical view.",
    "SENSITIVITY_ONLY_FLAG": "Indicator equal to 1 only for sensitivity-only analytical views that must never replace the canonical view.",
}

NOTES_OVERRIDES = {
    "GEST_AGE_CANONICAL": "Use this as the canonical gestational-age field; do not backfill from legacy scales downstream.",
    "PRETERM_FLAG": "Primary preterm endpoint. Traceable to the canonical event-level SINASC pathway.",
    "VERY_PRETERM_FLAG": "Sensitivity or severe-prematurity subset only. Must remain a strict subset of PRETERM_FLAG.",
    "GEST_DAYS_LOST_BELOW_37W": "Sensitivity metric only. Do not substitute this for PRETERM_FLAG in primary models.",
    "HEAT_INDEX_APPROX_C": "Approximate sensitivity exposure. Retain lineage fields and mixed-source provenance in analysis.",
    "DENOMINATOR_ZERO_FLAG": "When this equals 1, keep counts but do not compute rates or offsets.",
    "DENOMINATOR_MISSING_FLAG": "When this equals 1, keep counts but leave rates as NA and offsets invalid.",
    "RATE_NOT_COMPUTABLE_FLAG": "Use together with denominator flags to prevent pseudo-offset workarounds or log(0).",
    "OFFSET_VALID_FLAG": "Create model offsets only on rows where this equals 1.",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": "Retain this field when auditing station-level QC and ERA5 fallback after upstream quarantine.",
    "CLIMATE_RECORD_ANOMALY_FLAG": "Retain this field when auditing municipality-day anomaly correction and neighbor-based imputation.",
    "CLIMATE_CORRECTION_PATH": "Use together with the anomaly and quarantine flags to distinguish correction mechanisms in sensitivity analyses.",
    "SOURCE_DERIVED_VIEW": "Use to identify Stage 4 analytical views. Never treat the derived view as a silent replacement for canonical gold.",
    "VIEW_ROLE_NOTE": "Read this before comparing derived views with canonical tables; endpoint semantics remain canonical upstream.",
    "WRA_AGE_SCOPE_NOTE": "Use to distinguish canonical women 15-49 analyses from the 15-45 sensitivity view.",
    "CHILD_VIEW_AGE_LOGIC_NOTE": "Documents the explicit under-5 age split used by the child analytical view.",
    "SENSITIVITY_ONLY_FLAG": "Rows with value 1 belong to a sensitivity-only view and must never replace the canonical 15-49 view.",
}

SOURCE_PROVENANCE_OVERRIDES = {
    "GEST_AGE_CANONICAL": "Approved silver_sinasc_event canonical event pathway.",
    "PRETERM_FLAG": "Approved silver_sinasc_event PRETERM_LT37_FLAG aggregated to municipality-day.",
    "LOW_BIRTH_WEIGHT_LT2500_TOTAL": "Approved silver_sinasc_event birth-weight phenotype aggregated to municipality-day.",
    "PRETERM_LOW_BIRTH_WEIGHT_TOTAL": "Approved silver_sinasc_event intersected prematurity and low-birth-weight phenotype aggregated to municipality-day.",
    "PRENATAL_VISITS_8PLUS_FLAG": "Approved silver_sinasc_event exact-count prenatal 8+ phenotype aggregated to municipality-day.",
    "VERY_PRETERM_FLAG": "Approved silver_sinasc_event PRETERM_LT32_FLAG aggregated to municipality-day.",
    "GEST_DAYS_LOST_BELOW_37W": "Approved silver_sinasc_event canonical gestational-age derivation aggregated to municipality-day.",
    "PRENATAL_VISITS_COUNT": "Approved silver_sinasc_event prenatal fields retained as descriptor or sensitivity variables.",
    "PRENATAL_VISITS_6PLUS_FLAG": "Approved silver_sinasc_event prenatal fields retained as descriptor or sensitivity variables.",
    "PRENATAL_VISITS_SOURCE": "Approved silver_sinasc_event prenatal lineage fields.",
    "PRENATAL_VISITS_RAW_VALUE": "Approved silver_sinasc_event raw prenatal token retained for traceability.",
    "PRENATAL_VISITS_MISSING_FLAG": "Approved silver_sinasc_event prenatal missingness tracking.",
    "PRENATAL_VISITS_DERIVATION_NOTE": "Stage 1 perinatal derivation metadata from canonical SINASC aggregation.",
    "STILLBIRTH_PRIMARY_GTE28_FLAG": "Approved silver_stillbirth_individual primary >=28-week event flags.",
    "STILLBIRTH_SENS_GTE22_FLAG": "Approved silver_stillbirth_individual sensitivity >=22-week event flags.",
    "STILLBIRTH_TOTAL_PRIMARY_GTE28": "Approved silver_stillbirth_daily primary >=28-week stillbirth totals.",
    "STILLBIRTH_TOTAL_GTE22_SENSITIVITY": "Approved silver_stillbirth_daily >=22-week stillbirth sensitivity totals.",
    "TOTAL_BIRTHS_PRIMARY": "Approved silver_stillbirth_daily arithmetic: live births plus stillbirths >=28 weeks.",
    "TOTAL_BIRTHS_SENSITIVITY_GTE22": "Approved silver_stillbirth_daily arithmetic: live births plus stillbirths >=22 weeks.",
    "OBSTETRIC_ADMISSIONS_TOTAL": "Approved refined silver_sih_daily maternal phenotype block or approved derived WRA view built from canonical perinatal gold.",
    "HDP_ANY_TOTAL": "Approved refined silver_sih_daily maternal phenotype block or approved derived WRA view built from canonical perinatal gold.",
    "HDP_SEVERE_TOTAL": "Approved refined silver_sih_daily maternal phenotype block or approved derived WRA view built from canonical perinatal gold.",
    "HDP_GESTATIONAL_HTN_TOTAL": "Approved refined silver_sih_daily maternal phenotype block using the active ICD-10 dictionary.",
    "HDP_PREECLAMPSIA_TOTAL": "Approved refined silver_sih_daily maternal phenotype block using the active ICD-10 dictionary.",
    "HDP_ECLAMPSIA_TOTAL": "Approved refined silver_sih_daily maternal phenotype block using the active ICD-10 dictionary.",
    "PRETERM_LABOR_TOTAL": "Approved refined silver_sih_daily maternal phenotype block or approved derived WRA view built from canonical perinatal gold.",
    "APH_ANTEPARTUM_TOTAL": "Approved refined silver_sih_daily strict antepartum-haemorrhage phenotype or approved derived WRA view built from canonical perinatal gold.",
    "GDM_GESTATIONAL_TOTAL": "Approved refined silver_sih_daily maternal phenotype block or approved derived WRA view built from canonical perinatal gold.",
    "DEATHS_NEONATAL": "Approved silver_child_u5_sim_daily reconciled against silver_sim_individual and annual live births.",
    "DEATHS_U1": "Approved silver_child_u5_sim_daily reconciled against silver_sim_individual and annual live births.",
    "DEATHS_U5": "Approved silver_child_u5_sim_daily reconciled against silver_sim_individual and under-5 population.",
    "DEATHS_0_27D": "Approved Stage 4 child view derived directly from canonical gold_mnch_mortality.",
    "DEATHS_28_364D": "Approved Stage 4 child view derived directly from canonical gold_mnch_mortality.",
    "DEATHS_1_4Y": "Approved Stage 4 child view derived directly from canonical gold_mnch_mortality.",
    "DEATHS_UNDER5_TOTAL": "Approved Stage 4 child view derived directly from canonical gold_mnch_mortality.",
    "PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL": "Approved silver_child_u5_sih_daily pediatric admission endpoint retained in parallel to SIM mortality.",
    "PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL": "Approved silver_child_u5_sih_daily pediatric admission endpoint retained in parallel to SIM mortality.",
    "PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL": "Approved silver_child_u5_sih_daily pediatric admission endpoint retained in parallel to SIM mortality.",
    "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL": "Approved silver_child_u5_sih_daily pediatric admission sensitivity endpoint retained in parallel to SIM mortality.",
    "TMAX_C": "Approved climate_canonical_daily primary exposure or inherited approved gold climate enrichment.",
    "TMIN_C": "Approved climate_canonical_daily temperature exposure or inherited approved gold climate enrichment.",
    "TMEAN_C": "Approved climate_canonical_daily temperature exposure or inherited approved gold climate enrichment.",
    "RH_MIN_PCT": "Approved climate_canonical_daily humidity co-exposure or inherited approved gold climate enrichment.",
    "RH_MEAN_PCT": "Approved climate_canonical_daily humidity co-exposure or inherited approved gold climate enrichment.",
    "HEATWAVE_LEN_P95": "Approved climate_canonical_daily duration-sensitive sensitivity exposure or inherited approved gold climate enrichment.",
    "HEAT_INDEX_APPROX_C": "Approved derived heat-index sensitivity exposure from the final merged climate series.",
    "CLIMATE_UPSTREAM_QUARANTINE_FLAG": "Approved climate_canonical_daily upstream correction provenance retained through gold climate enrichment.",
    "CLIMATE_RECORD_ANOMALY_FLAG": "Approved climate_canonical_daily anomaly correction provenance retained through gold climate enrichment.",
    "CLIMATE_CORRECTION_PATH": "Approved climate_canonical_daily compact correction-path lineage retained through gold climate enrichment.",
    "ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL": "Approved child mortality cause-group block with explicit X30 any-mention retention.",
}

DERIVED_DESCRIPTION_OVERRIDES = {
    ("gold_analysis_child_u5", "DEATHS_0_27D"): "Daily deaths aged 0-27 days derived from canonical gold_mnch_mortality.",
    ("gold_analysis_child_u5", "DEATHS_28_364D"): "Daily deaths aged 28-364 days derived from canonical gold_mnch_mortality.",
    ("gold_analysis_child_u5", "DEATHS_1_4Y"): "Daily deaths aged 1-4 completed years derived from canonical gold_mnch_mortality.",
    ("gold_analysis_child_u5", "DEATHS_UNDER5_TOTAL"): "Daily under-five deaths derived from canonical gold_mnch_mortality.",
    ("gold_analysis_child_u5", "CHILD_VIEW_AGE_LOGIC_NOTE"): "Documentation note recording the explicit 0-27d, 28-364d, 1-4y, and under-5 age logic.",
    ("gold_analysis_child_u5", "VIEW_ROLE_NOTE"): "Documentation note stating that the child view is analytical only and does not replace canonical mortality gold.",
    ("gold_analysis_child_u5", "SOURCE_DERIVED_VIEW"): "Lineage marker showing that the record was emitted by the Stage 4 child analytical view.",
    ("gold_analysis_wra_15_49", "POP_WRA_15_49"): "Annual women 15-49 population denominator carried into the canonical WRA analytical view.",
    ("gold_analysis_wra_15_49", "OFFSET_LOG_POP_WRA_15_49"): "Log of the annual women 15-49 population denominator when positive.",
    ("gold_analysis_wra_15_49", "SOURCE_CANONICAL_GOLD"): "Lineage marker showing that the view was derived from canonical gold_mnch_perinatal.",
    ("gold_analysis_wra_15_49", "VIEW_ROLE_NOTE"): "Documentation note stating that the WRA 15-49 view is analytical only and does not replace canonical perinatal gold.",
    ("gold_analysis_wra_15_49", "WRA_AGE_SCOPE_NOTE"): "Documentation note stating that this is the canonical women 15-49 analytical view.",
    ("gold_analysis_wra_15_49", "SENSITIVITY_ONLY_FLAG"): "Indicator set to 0 because this is the canonical, not sensitivity-only, WRA analytical view.",
    ("gold_analysis_wra_15_49", "SOURCE_DERIVED_VIEW"): "Lineage marker showing that the record was emitted by the Stage 4 women 15-49 analytical view.",
    ("gold_analysis_wra_15_45_sens", "POP_WRA_15_45_SENSITIVITY"): "Annual women 15-45 sensitivity population denominator carried into the sensitivity-only WRA analytical view.",
    ("gold_analysis_wra_15_45_sens", "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY"): "Log of the annual women 15-45 sensitivity population denominator when positive.",
    ("gold_analysis_wra_15_45_sens", "SOURCE_CANONICAL_GOLD"): "Lineage marker showing that the sensitivity view was derived from canonical gold_mnch_perinatal.",
    ("gold_analysis_wra_15_45_sens", "VIEW_ROLE_NOTE"): "Documentation note stating that the WRA 15-45 view is analytical and sensitivity-only.",
    ("gold_analysis_wra_15_45_sens", "WRA_AGE_SCOPE_NOTE"): "Documentation note stating that this is the women 15-45 sensitivity analytical view.",
    ("gold_analysis_wra_15_45_sens", "SENSITIVITY_ONLY_FLAG"): "Indicator set to 1 because this view is sensitivity-only and must not replace the canonical 15-49 view.",
    ("gold_analysis_wra_15_45_sens", "SOURCE_DERIVED_VIEW"): "Lineage marker showing that the record was emitted by the Stage 4 women 15-45 sensitivity view.",
}

COMPATIBILITY_NOTES = {
    "STILLBIRTH_TOTAL": "Compatibility alias retained for the primary stillbirth total; prefer STILLBIRTH_TOTAL_PRIMARY_GTE28 in new analysis.",
    "NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS": "Compatibility alias retained for total births under the primary stillbirth definition; prefer TOTAL_BIRTHS_PRIMARY.",
    "OFFSET_LOG_NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS": "Compatibility alias retained for the primary stillbirth log offset; prefer OFFSET_LOG_TOTAL_BIRTHS_PRIMARY.",
    "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS": "Compatibility alias retained for the primary stillbirth rate; prefer STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY.",
    "NEONATAL_DEATHS_TOTAL": "Compatibility alias retained for DEATHS_NEONATAL; prefer the canonical DEATHS_NEONATAL field.",
    "INFANT_DEATHS_TOTAL": "Compatibility alias retained for DEATHS_U1; prefer the canonical DEATHS_U1 field.",
    "UNDER5_DEATHS_TOTAL": "Compatibility alias retained for DEATHS_U5; prefer the canonical DEATHS_U5 field.",
    "LIVE_BIRTHS_TOTAL": "Compatibility alias retained where annual live births are carried under a legacy-friendly name.",
    "UNDER5_POPULATION": "Compatibility alias retained for POP_U5; prefer POP_U5 in new analysis.",
    "CRUDE_RATE_NEONATAL": "Compatibility alias retained for NEONATAL_MORTALITY_RATE_PER_1000_LIVE_BIRTHS.",
    "CRUDE_RATE_U1": "Compatibility alias retained for INFANT_MORTALITY_RATE_PER_1000_LIVE_BIRTHS.",
    "CRUDE_RATE_U5": "Compatibility alias retained for UNDER5_MORTALITY_RATE_PER_100000_UNDER5_POPULATION.",
}


@dataclass(frozen=True)
class EndpointSpec:
    endpoint_name: str
    gold_table: str
    canonical_source: str
    required_fields: tuple[str, ...]
    denominator_rule: str
    denominator_fields: tuple[str, ...]
    climate_required: bool = True
    age_scope: str = ""
    primary_or_sensitivity: str = "PRIMARY"


ENDPOINT_SPECS = (
    EndpointSpec(
        endpoint_name="PRETERM_FLAG",
        gold_table="gold_mnch_perinatal",
        canonical_source="silver_sinasc_event -> gold_mnch_perinatal",
        required_fields=("PRETERM_FLAG", "GEST_AGE_CANONICAL", "LIVE_BIRTHS_TOTAL"),
        denominator_rule="daily live births for descriptive rates; no canonical gold-side age truncation",
        denominator_fields=("LIVE_BIRTHS_TOTAL",),
        age_scope="all births / no age truncation",
    ),
    EndpointSpec(
        endpoint_name="LOW_BIRTH_WEIGHT_LT2500_TOTAL",
        gold_table="gold_mnch_perinatal",
        canonical_source="silver_sinasc_event -> gold_mnch_perinatal",
        required_fields=("LOW_BIRTH_WEIGHT_LT2500_TOTAL", "BIRTH_WEIGHT_REPORTED_TOTAL", "LIVE_BIRTHS_TOTAL"),
        denominator_rule="daily live births with explicit birth-weight capture retained for completeness auditing",
        denominator_fields=("LIVE_BIRTHS_TOTAL",),
        age_scope="all births / no age truncation",
    ),
    EndpointSpec(
        endpoint_name="PRETERM_LOW_BIRTH_WEIGHT_TOTAL",
        gold_table="gold_mnch_perinatal",
        canonical_source="silver_sinasc_event -> gold_mnch_perinatal",
        required_fields=("PRETERM_LOW_BIRTH_WEIGHT_TOTAL", "LOW_BIRTH_WEIGHT_LT2500_TOTAL", "PRETERM_FLAG"),
        denominator_rule="daily live births; intersected phenotype must remain a subset of PRETERM_FLAG",
        denominator_fields=("LIVE_BIRTHS_TOTAL",),
        age_scope="all births / no age truncation",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="VERY_PRETERM_FLAG",
        gold_table="gold_mnch_perinatal",
        canonical_source="silver_sinasc_event -> gold_mnch_perinatal",
        required_fields=("VERY_PRETERM_FLAG", "PRETERM_FLAG", "GEST_AGE_CANONICAL"),
        denominator_rule="daily live births for descriptive rates; strict subset of PRETERM_FLAG",
        denominator_fields=("LIVE_BIRTHS_TOTAL",),
        age_scope="all births / no age truncation",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="GEST_DAYS_LOST_BELOW_37W",
        gold_table="gold_mnch_perinatal",
        canonical_source="silver_sinasc_event -> gold_mnch_perinatal",
        required_fields=("GEST_DAYS_LOST_BELOW_37W", "GEST_AGE_CANONICAL"),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="all births / no age truncation",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="STILLBIRTH_TOTAL_PRIMARY_GTE28",
        gold_table="gold_mnch_perinatal",
        canonical_source="silver_stillbirth_daily + silver_stillbirth_individual -> gold_mnch_perinatal",
        required_fields=("STILLBIRTH_TOTAL_PRIMARY_GTE28", "TOTAL_BIRTHS_PRIMARY", "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"),
        denominator_rule="daily total births primary = live births + stillbirths >=28 weeks; never women 15-49",
        denominator_fields=("TOTAL_BIRTHS_PRIMARY", "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY"),
        age_scope="all births / no age truncation",
    ),
    EndpointSpec(
        endpoint_name="STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
        gold_table="gold_mnch_perinatal",
        canonical_source="silver_stillbirth_daily + silver_stillbirth_individual -> gold_mnch_perinatal",
        required_fields=("STILLBIRTH_TOTAL_GTE22_SENSITIVITY", "TOTAL_BIRTHS_SENSITIVITY_GTE22", "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22"),
        denominator_rule="daily total births sensitivity = live births + stillbirths >=22 weeks; never women 15-49",
        denominator_fields=("TOTAL_BIRTHS_SENSITIVITY_GTE22", "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22"),
        age_scope="all births / no age truncation",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="DEATHS_NEONATAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + silver_sinasc_daily -> gold_mnch_mortality",
        required_fields=("DEATHS_NEONATAL", "LIVE_BIRTHS_TOTAL_ANNUAL", "OFFSET_LOG_LIVE_BIRTHS"),
        denominator_rule="annual live births",
        denominator_fields=("LIVE_BIRTHS_TOTAL_ANNUAL", "OFFSET_LOG_LIVE_BIRTHS"),
        age_scope="0-27 days",
    ),
    EndpointSpec(
        endpoint_name="DEATHS_U1",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + silver_sinasc_daily -> gold_mnch_mortality",
        required_fields=("DEATHS_U1", "LIVE_BIRTHS_TOTAL_ANNUAL", "OFFSET_LOG_LIVE_BIRTHS"),
        denominator_rule="annual live births",
        denominator_fields=("LIVE_BIRTHS_TOTAL_ANNUAL", "OFFSET_LOG_LIVE_BIRTHS"),
        age_scope="0-364 days",
    ),
    EndpointSpec(
        endpoint_name="DEATHS_U5",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + silver_pop_under5 -> gold_mnch_mortality",
        required_fields=("DEATHS_U5", "POP_U5", "OFFSET_LOG_UNDER5_POPULATION"),
        denominator_rule="under-5 population",
        denominator_fields=("POP_U5", "OFFSET_LOG_UNDER5_POPULATION"),
        age_scope="0-59 months / 0-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + active ICD-10 dictionary -> gold_mnch_mortality",
        required_fields=("UNDERLYING_BASIC_CAUSE_DIRECT_HEAT_ANY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="ANY_MENTION_DIRECT_HEAT_ANY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + active ICD-10 dictionary -> gold_mnch_mortality",
        required_fields=("ANY_MENTION_DIRECT_HEAT_ANY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + active ICD-10 dictionary -> gold_mnch_mortality",
        required_fields=("UNDERLYING_BASIC_CAUSE_INDIRECT_HEAT_ANY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + active ICD-10 dictionary -> gold_mnch_mortality",
        required_fields=("ANY_MENTION_INDIRECT_HEAT_ANY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sim_daily + silver_sim_individual + active ICD-10 dictionary -> gold_mnch_mortality",
        required_fields=("ANY_MENTION_EXCESSIVE_NATURAL_HEAT_X30_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
        primary_or_sensitivity="PRIMARY",
    ),
    EndpointSpec(
        endpoint_name="ACUTE_CARE_PROXY_TOTAL",
        gold_table="gold_mnch_panel",
        canonical_source="gold_mnch_mortality + gold_mnch_perinatal + silver_pop_wra_15_49 -> gold_mnch_panel",
        required_fields=("ACUTE_CARE_PROXY_TOTAL", "ACUTE_CARE_PROXY_LABEL"),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="mixed municipality-day comparative mart",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sih_daily -> gold_mnch_mortality",
        required_fields=("PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sih_daily -> gold_mnch_mortality",
        required_fields=("PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sih_daily -> gold_mnch_mortality",
        required_fields=("PRINCIPAL_DIAGNOSIS_CHILD_RESPIRATORY_ACUTE_PRIMARY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",
        gold_table="gold_mnch_mortality",
        canonical_source="silver_child_u5_sih_daily -> gold_mnch_mortality",
        required_fields=("ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL",),
        denominator_rule=NOT_APPLICABLE_DENOMINATOR,
        denominator_fields=(),
        age_scope="0-59 months / 0-4 completed years",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="DEATHS_0_27D",
        gold_table="gold_analysis_child_u5",
        canonical_source="gold_mnch_mortality -> gold_analysis_child_u5",
        required_fields=("DEATHS_0_27D", "LIVE_BIRTHS_TOTAL_ANNUAL", "CHILD_VIEW_AGE_LOGIC_NOTE"),
        denominator_rule="annual live births",
        denominator_fields=("LIVE_BIRTHS_TOTAL_ANNUAL",),
        age_scope="0-27 days",
    ),
    EndpointSpec(
        endpoint_name="DEATHS_28_364D",
        gold_table="gold_analysis_child_u5",
        canonical_source="gold_mnch_mortality -> gold_analysis_child_u5",
        required_fields=("DEATHS_28_364D", "LIVE_BIRTHS_TOTAL_ANNUAL", "CHILD_VIEW_AGE_LOGIC_NOTE"),
        denominator_rule="annual live births",
        denominator_fields=("LIVE_BIRTHS_TOTAL_ANNUAL",),
        age_scope="28-364 days",
    ),
    EndpointSpec(
        endpoint_name="DEATHS_1_4Y",
        gold_table="gold_analysis_child_u5",
        canonical_source="gold_mnch_mortality -> gold_analysis_child_u5",
        required_fields=("DEATHS_1_4Y", "POP_U5", "CHILD_VIEW_AGE_LOGIC_NOTE"),
        denominator_rule="under-5 population",
        denominator_fields=("POP_U5",),
        age_scope="1-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="DEATHS_UNDER5_TOTAL",
        gold_table="gold_analysis_child_u5",
        canonical_source="gold_mnch_mortality -> gold_analysis_child_u5",
        required_fields=("DEATHS_UNDER5_TOTAL", "POP_U5", "CHILD_VIEW_AGE_LOGIC_NOTE"),
        denominator_rule="under-5 population",
        denominator_fields=("POP_U5",),
        age_scope="0-59 months / 0-4 completed years",
    ),
    EndpointSpec(
        endpoint_name="OBSTETRIC_ADMISSIONS_TOTAL_WRA_15_49",
        gold_table="gold_analysis_wra_15_49",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_49 -> gold_analysis_wra_15_49",
        required_fields=("OBSTETRIC_ADMISSIONS_TOTAL", "POP_WRA_15_49", "WRA_AGE_SCOPE_NOTE"),
        denominator_rule="women 15-49 population",
        denominator_fields=("POP_WRA_15_49", "OFFSET_LOG_POP_WRA_15_49"),
        age_scope="women 15-49 canonical",
    ),
    EndpointSpec(
        endpoint_name="HDP_ANY_TOTAL_WRA_15_49",
        gold_table="gold_analysis_wra_15_49",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_49 -> gold_analysis_wra_15_49",
        required_fields=("HDP_ANY_TOTAL", "POP_WRA_15_49", "WRA_AGE_SCOPE_NOTE"),
        denominator_rule="women 15-49 population",
        denominator_fields=("POP_WRA_15_49", "OFFSET_LOG_POP_WRA_15_49"),
        age_scope="women 15-49 canonical",
    ),
    EndpointSpec(
        endpoint_name="PRETERM_LABOR_TOTAL_WRA_15_49",
        gold_table="gold_analysis_wra_15_49",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_49 -> gold_analysis_wra_15_49",
        required_fields=("PRETERM_LABOR_TOTAL", "POP_WRA_15_49", "WRA_AGE_SCOPE_NOTE"),
        denominator_rule="women 15-49 population",
        denominator_fields=("POP_WRA_15_49", "OFFSET_LOG_POP_WRA_15_49"),
        age_scope="women 15-49 canonical",
    ),
    EndpointSpec(
        endpoint_name="GDM_GESTATIONAL_TOTAL_WRA_15_49",
        gold_table="gold_analysis_wra_15_49",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_49 -> gold_analysis_wra_15_49",
        required_fields=("GDM_GESTATIONAL_TOTAL", "POP_WRA_15_49", "WRA_AGE_SCOPE_NOTE"),
        denominator_rule="women 15-49 population",
        denominator_fields=("POP_WRA_15_49", "OFFSET_LOG_POP_WRA_15_49"),
        age_scope="women 15-49 canonical",
    ),
    EndpointSpec(
        endpoint_name="APH_ANTEPARTUM_TOTAL_WRA_15_49",
        gold_table="gold_analysis_wra_15_49",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_49 -> gold_analysis_wra_15_49",
        required_fields=("APH_ANTEPARTUM_TOTAL", "POP_WRA_15_49", "WRA_AGE_SCOPE_NOTE"),
        denominator_rule="women 15-49 population",
        denominator_fields=("POP_WRA_15_49", "OFFSET_LOG_POP_WRA_15_49"),
        age_scope="women 15-49 canonical",
    ),
    EndpointSpec(
        endpoint_name="OBSTETRIC_ADMISSIONS_TOTAL_WRA_15_45_SENS",
        gold_table="gold_analysis_wra_15_45_sens",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_45_sensitivity -> gold_analysis_wra_15_45_sens",
        required_fields=("OBSTETRIC_ADMISSIONS_TOTAL", "POP_WRA_15_45_SENSITIVITY", "WRA_AGE_SCOPE_NOTE", "SENSITIVITY_ONLY_FLAG"),
        denominator_rule="women 15-45 sensitivity population",
        denominator_fields=("POP_WRA_15_45_SENSITIVITY", "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY"),
        age_scope="women 15-45 sensitivity",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="HDP_ANY_TOTAL_WRA_15_45_SENS",
        gold_table="gold_analysis_wra_15_45_sens",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_45_sensitivity -> gold_analysis_wra_15_45_sens",
        required_fields=("HDP_ANY_TOTAL", "POP_WRA_15_45_SENSITIVITY", "WRA_AGE_SCOPE_NOTE", "SENSITIVITY_ONLY_FLAG"),
        denominator_rule="women 15-45 sensitivity population",
        denominator_fields=("POP_WRA_15_45_SENSITIVITY", "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY"),
        age_scope="women 15-45 sensitivity",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="PRETERM_LABOR_TOTAL_WRA_15_45_SENS",
        gold_table="gold_analysis_wra_15_45_sens",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_45_sensitivity -> gold_analysis_wra_15_45_sens",
        required_fields=("PRETERM_LABOR_TOTAL", "POP_WRA_15_45_SENSITIVITY", "WRA_AGE_SCOPE_NOTE", "SENSITIVITY_ONLY_FLAG"),
        denominator_rule="women 15-45 sensitivity population",
        denominator_fields=("POP_WRA_15_45_SENSITIVITY", "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY"),
        age_scope="women 15-45 sensitivity",
        primary_or_sensitivity="SENSITIVITY",
    ),
    EndpointSpec(
        endpoint_name="GDM_GESTATIONAL_TOTAL_WRA_15_45_SENS",
        gold_table="gold_analysis_wra_15_45_sens",
        canonical_source="gold_mnch_perinatal + silver_pop_wra_15_45_sensitivity -> gold_analysis_wra_15_45_sens",
        required_fields=("GDM_GESTATIONAL_TOTAL", "POP_WRA_15_45_SENSITIVITY", "WRA_AGE_SCOPE_NOTE", "SENSITIVITY_ONLY_FLAG"),
        denominator_rule="women 15-45 sensitivity population",
        denominator_fields=("POP_WRA_15_45_SENSITIVITY", "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY"),
        age_scope="women 15-45 sensitivity",
        primary_or_sensitivity="SENSITIVITY",
    ),
)


def _gold_output_path(root: Path, table: str, year: int) -> Path:
    return root / "data" / "gold" / table / f"YEAR={year}" / f"{table}_{year}.parquet"


def _stage_mini_report_path(root: Path, table: str) -> Path:
    if table == "gold_mnch_perinatal":
        return root / "artifacts" / "status" / "gold_stage1_perinatal_mini_report.csv"
    if table == "gold_mnch_mortality":
        return root / "artifacts" / "status" / "gold_stage2_mortality_mini_report.csv"
    if table == "gold_mnch_panel":
        return root / "artifacts" / "status" / "gold_stage3_panel_mini_report.csv"
    return root / "artifacts" / "status" / "gold_stage4_derived_mini_report.csv"


def _stage_key_column(table: str) -> str:
    return "table" if table in CANONICAL_GOLD_TABLES else "view"


def _load_stage_evidence(root: Path, table: str) -> dict[int, dict[str, Any]]:
    path = _stage_mini_report_path(root, table)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    key_column = _stage_key_column(table)
    table_rows = frame.loc[frame[key_column].astype(str) == table].copy()
    evidence: dict[int, dict[str, Any]] = {}
    for row in table_rows.to_dict(orient="records"):
        evidence[int(row["year"])] = row
    return evidence


def _schema_description_map(root: Path, table: str) -> dict[str, str]:
    schema_name = TABLE_SCHEMA_FILES.get(table)
    if not schema_name:
        return {}
    path = root / "schemas" / schema_name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row["column_name"]): str(row.get("description", "")).strip()
            for row in reader
            if row.get("column_name")
        }


def _canonical_description_index(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for table in CANONICAL_GOLD_TABLES:
        for column, description in _schema_description_map(root, table).items():
            index.setdefault(column, description)
    return index


def _humanize_column_name(column: str) -> str:
    tokens = column.split("_")
    replacements = {
        "U5": "under-5",
        "U1": "under-1",
        "WRA": "WRA",
        "ICU": "ICU",
        "HDP": "HDP",
        "GDM": "GDM",
        "TMAX": "TMAX",
        "RH": "RH",
        "SIM": "SIM",
        "SIH": "SIH",
        "SINASC": "SINASC",
        "QC": "QC",
        "GTE28": ">=28w",
        "GTE22": ">=22w",
        "P95": "P95",
    }
    rendered = [replacements.get(token, token.lower()) for token in tokens if token]
    return " ".join(rendered).replace("  ", " ").strip().capitalize()


def _plain_label(column: str, definition: str) -> str:
    if column in PLAIN_LABEL_OVERRIDES:
        return PLAIN_LABEL_OVERRIDES[column]
    if definition:
        return definition.rstrip(".")
    return _humanize_column_name(column)


def _inherited_definition(root: Path, table: str, column: str, canonical_descriptions: dict[str, str]) -> str:
    if (table, column) in DERIVED_DESCRIPTION_OVERRIDES:
        return DERIVED_DESCRIPTION_OVERRIDES[(table, column)]
    if column in DEFINITION_OVERRIDES:
        return DEFINITION_OVERRIDES[column]
    inherited = canonical_descriptions.get(column, "")
    if inherited:
        return inherited
    return f"{_humanize_column_name(column)} in {table}."


def _unit(column: str) -> str:
    if column == "DATE":
        return "ISO date"
    if column == "YEAR":
        return "calendar year"
    if column == "MONTH":
        return "month number"
    if column.endswith("_FLAG"):
        return "0/1 indicator"
    if column in {"TMAX_C", "TMIN_C", "TMEAN_C", "HEAT_INDEX_APPROX_C"}:
        return "degrees Celsius"
    if column in {"RH_MIN_PCT", "RH_MEAN_PCT"}:
        return "percent"
    if "RATE_PER_1000" in column:
        return "rate per 1,000"
    if "RATE_PER_100000" in column:
        return "rate per 100,000"
    if "OFFSET_LOG" in column:
        return "log(denominator)"
    if column.startswith("OFFSET_VALUE"):
        return "count"
    if column.startswith("POP_") or column.startswith("REF_POP_"):
        return "population count"
    if "LIVE_BIRTHS" in column or "TOTAL_BIRTHS" in column or column.endswith("_TOTAL") or column.startswith("NV_"):
        return "count"
    if "DAYS" in column or "LEN" in column:
        return "days"
    if "WEEKS" in column:
        return "weeks"
    if "PCT" in column:
        return "percent"
    if column.endswith("_NOTE") or column.endswith("_LABEL") or column.startswith("SOURCE_") or column.startswith("FONTE_") or column == "CLIMATE_CORRECTION_PATH":
        return "text"
    return "value"


def _key_role(column: str) -> str:
    if column == "MUNICIPALITY_CODE":
        return "primary municipality key"
    if column == "DATE":
        return "primary date key"
    if column in {"YEAR", "MONTH"}:
        return "calendar helper key"
    if column.startswith("SOURCE_") or column.startswith("FONTE_") or column.startswith("CLIMATE_"):
        return "lineage marker"
    if column.endswith("_NOTE") or column.endswith("_LABEL"):
        return "documentation helper"
    return "measure"


def _source_provenance(table: str, column: str) -> str:
    if column in SOURCE_PROVENANCE_OVERRIDES:
        return SOURCE_PROVENANCE_OVERRIDES[column]
    if column.startswith("SOURCE_") or column.startswith("FONTE_") or column.startswith("IMPUTADO_") or column.startswith("CLIMATE_") or column in {
        "HEAT_INDEX_METHOD",
        "HEAT_INDEX_SOURCE_PAIR",
        "HEAT_INDEX_INPUTS_ZERO_MISSING",
    }:
        return "Lineage field carried from the approved upstream stage into the final gold artifact."
    return TABLE_SOURCE_PROVENANCE[table]


def _primary_or_sensitivity(column: str) -> str:
    if column in {"TMAX_C", "TMIN_C", "TMEAN_C"}:
        return "PRIMARY"
    if column in SECONDARY_OR_SENSITIVITY_FIELDS or "SENSITIVITY" in column or column.endswith("_APPROX_C"):
        return "SENSITIVITY"
    if column.startswith("SOURCE_") or column.startswith("FONTE_") or column.startswith("IMPUTADO_") or column.startswith("CLIMATE_") or column.endswith("_NOTE") or column.endswith("_LABEL"):
        return "DESCRIPTOR"
    if column in LEGACY_COMPATIBILITY_FIELDS:
        return "DESCRIPTOR"
    return "PRIMARY"


def _expected_missingness(table: str, column: str) -> str:
    if column in {"TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "HEATWAVE_LEN_P95", "HEAT_INDEX_APPROX_C"}:
        if table in {"gold_mnch_perinatal", "gold_mnch_panel", "gold_analysis_wra_15_49", "gold_analysis_wra_15_45_sens"}:
            return "rare only on explicit no-climate-backbone-match rows"
        return "none expected after approved build"
    if column in {"PRENATAL_VISITS_COUNT", "PRENATAL_VISITS_RAW_VALUE"}:
        return "source-driven missingness allowed"
    if column in {"PRENATAL_VISITS_MISSING_FLAG", "GEST_AGE_MISSING_FLAG"}:
        return "none expected"
    if "OFFSET_LOG" in column or "RATE" in column or "OFFSET_VALUE" in column:
        return "missing when denominator is invalid or not applicable"
    if column == "OFFSET_VALUE":
        return "missing by design in the wide mortality mart"
    if column.startswith("SOURCE_") or column.startswith("FONTE_") or column.startswith("CLIMATE_") or column.endswith("_NOTE") or column.endswith("_LABEL") or column == "QC_STATUS":
        return "none expected"
    return "none expected"


def _outcome_family(table: str, column: str) -> str:
    if column in {"MUNICIPALITY_CODE", "DATE", "YEAR", "MONTH"}:
        return "keys"
    if column in {"TMAX_C", "TMIN_C", "TMEAN_C", "RH_MIN_PCT", "RH_MEAN_PCT", "HEATWAVE_LEN_P95", "HEAT_INDEX_APPROX_C"} or column.startswith("FONTE_") or column.startswith("IMPUTADO_"):
        return "climate"
    if column.startswith("CLIMATE_"):
        return "provenance_and_qc"
    if column.startswith("SOURCE_") or column.endswith("_NOTE") or column.endswith("_LABEL") or column == "QC_STATUS":
        return "provenance_and_qc"
    if column.startswith("POP_") or column.startswith("REF_") or "DENOMINATOR" in column or column.startswith("OFFSET") or "RATE_" in column or column.startswith("CRUDE_RATE"):
        return "denominator_and_rate"
    if column in MATERNAL_PHENOTYPE_FIELDS:
        return "maternal_service"
    if column.startswith("PRENATAL_"):
        return "prenatal_descriptor"
    if column.startswith("STILLBIRTH") or "TOTAL_BIRTHS" in column or column.startswith("NASCIMENTOS_TOTAIS"):
        return "stillbirth"
    if column.startswith("GEST_") or column.startswith("NV_") or column in {"PRETERM_FLAG", "VERY_PRETERM_FLAG"}:
        return "perinatal"
    if column.startswith("DEATHS_") or "MORTALITY" in column or "UNDERLYING_BASIC_CAUSE" in column or "ANY_MENTION" in column:
        return "child_mortality"
    if table.startswith("gold_analysis_wra_"):
        return "maternal_service"
    return "gold_delivery"


def _age_scope(table: str, column: str) -> str:
    if column in {"DEATHS_NEONATAL", "DEATHS_0_27D"}:
        return "0-27 days"
    if column in {"DEATHS_U1", "DEATHS_28_364D"}:
        return "0-364 days" if column == "DEATHS_U1" else "28-364 days"
    if column == "DEATHS_1_4Y":
        return "1-4 completed years"
    if column in {
        "DEATHS_U5",
        "UNDER5_DEATHS_TOTAL",
        "DEATHS_UNDER5_TOTAL",
        "POP_U5",
        "UNDER5_POPULATION",
        "UNDER5_MORTALITY_RATE_PER_100000_UNDER5_POPULATION",
    }:
        return "0-59 months / 0-4 completed years"
    if table.startswith("gold_analysis_wra_15_49"):
        return "women 15-49 canonical"
    if table.startswith("gold_analysis_wra_15_45_sens"):
        return "women 15-45 sensitivity"
    return TABLE_AGE_SCOPE[table]


def _denominator_rule(table: str, column: str) -> str:
    if column in {"PRETERM_FLAG", "VERY_PRETERM_FLAG", "NV_PRETERM_TOTAL", "PRETERM_BIRTHS_CANONICAL_TOTAL"}:
        return "daily live births when descriptive rates are needed"
    if column in {
        "STILLBIRTH_PRIMARY_GTE28_FLAG",
        "STILLBIRTH_TOTAL_PRIMARY_GTE28",
        "TOTAL_BIRTHS_PRIMARY",
        "OFFSET_LOG_TOTAL_BIRTHS_PRIMARY",
        "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_PRIMARY",
        "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS",
        "STILLBIRTH_TOTAL",
    }:
        return "daily total births primary = live births + stillbirths >=28 weeks"
    if column in {
        "STILLBIRTH_SENS_GTE22_FLAG",
        "STILLBIRTH_TOTAL_GTE22_SENSITIVITY",
        "TOTAL_BIRTHS_SENSITIVITY_GTE22",
        "OFFSET_LOG_TOTAL_BIRTHS_SENSITIVITY_GTE22",
        "STILLBIRTH_RATE_PER_1000_TOTAL_BIRTHS_SENSITIVITY_GTE22",
    }:
        return "daily total births sensitivity = live births + stillbirths >=22 weeks"
    if column in {
        "DEATHS_NEONATAL",
        "DEATHS_U1",
        "NEONATAL_DEATHS_TOTAL",
        "INFANT_DEATHS_TOTAL",
        "LIVE_BIRTHS_TOTAL_ANNUAL",
        "OFFSET_LOG_LIVE_BIRTHS",
        "NEONATAL_MORTALITY_RATE_PER_1000_LIVE_BIRTHS",
        "INFANT_MORTALITY_RATE_PER_1000_LIVE_BIRTHS",
        "CRUDE_RATE_NEONATAL",
        "CRUDE_RATE_U1",
        "DEATHS_0_27D",
        "DEATHS_28_364D",
    }:
        return "annual live births"
    if column in {
        "DEATHS_U5",
        "UNDER5_DEATHS_TOTAL",
        "DEATHS_1_4Y",
        "DEATHS_UNDER5_TOTAL",
        "POP_U5",
        "UNDER5_POPULATION",
        "OFFSET_LOG_UNDER5_POPULATION",
        "UNDER5_MORTALITY_RATE_PER_100000_UNDER5_POPULATION",
        "CRUDE_RATE_U5",
    }:
        return "under-5 population"
    if column in {
        "OBSTETRIC_ADMISSIONS_TOTAL",
        "HDP_ANY_TOTAL",
        "HDP_SEVERE_TOTAL",
        "HDP_GESTATIONAL_HTN_TOTAL",
        "HDP_PREECLAMPSIA_TOTAL",
        "HDP_ECLAMPSIA_TOTAL",
        "PRETERM_LABOR_TOTAL",
        "GDM_GESTATIONAL_TOTAL",
        "POP_WRA_15_49",
        "OFFSET_LOG_POP_WRA_15_49",
    } and table == "gold_analysis_wra_15_49":
        return "women 15-49 population"
    if column in {
        "OBSTETRIC_ADMISSIONS_TOTAL",
        "HDP_ANY_TOTAL",
        "HDP_SEVERE_TOTAL",
        "HDP_GESTATIONAL_HTN_TOTAL",
        "HDP_PREECLAMPSIA_TOTAL",
        "HDP_ECLAMPSIA_TOTAL",
        "PRETERM_LABOR_TOTAL",
        "GDM_GESTATIONAL_TOTAL",
        "POP_WRA_15_45_SENSITIVITY",
        "OFFSET_LOG_POP_WRA_15_45_SENSITIVITY",
    } and table == "gold_analysis_wra_15_45_sens":
        return "women 15-45 sensitivity population"
    if column.startswith("REF_LIVE_BIRTHS_DAILY"):
        return "daily live births"
    if column.startswith("REF_TOTAL_BIRTHS_PRIMARY_DAILY"):
        return "daily total births primary = live births + stillbirths >=28 weeks"
    if column.startswith("REF_LIVE_BIRTHS_TOTAL_ANNUAL"):
        return "annual live births"
    if column.startswith("REF_POP_U5"):
        return "under-5 population"
    if column.startswith("REF_POP_WRA_15_49"):
        return "women 15-49 population"
    if "DENOMINATOR" in column or column.startswith("OFFSET") or column.startswith("RATE_") or column.startswith("CRUDE_RATE"):
        return "inherits the endpoint-specific denominator contract documented in the sibling measure fields"
    return NOT_APPLICABLE_DENOMINATOR


def _canonical_or_legacy(column: str) -> str:
    if column in LEGACY_COMPATIBILITY_FIELDS:
        return "legacy_compatibility_retained"
    if column in {"DIABETES_PREGNANCY_BROAD_TOTAL", "HEAT_INDEX_APPROX_C", "HEATWAVE_LEN_P95", "VERY_PRETERM_FLAG", "GEST_DAYS_LOST_BELOW_37W", "ANY_DIAGNOSIS_CHILD_ASTHMA_WHEEZE_SENSITIVITY_TOTAL"}:
        return "canonical_sensitivity_retained"
    return "canonical"


def _notes_on_use(table: str, column: str) -> str:
    if column in NOTES_OVERRIDES:
        return NOTES_OVERRIDES[column]
    if column in COMPATIBILITY_NOTES:
        return COMPATIBILITY_NOTES[column]
    if column.startswith("SOURCE_") or column.startswith("FONTE_") or column.startswith("IMPUTADO_"):
        return "Preserve for provenance and fallback tracing in any downstream analytical output."
    if column.startswith("OFFSET_LOG") or column == "OFFSET_VALUE":
        return "Use only when the matching denominator is positive and the relevant offset-valid flag permits modeling."
    if "RATE" in column:
        return "Descriptive rate only. Final models should use counts plus valid offsets rather than this precomputed rate."
    if column.endswith("_FLAG"):
        return "Binary or count-style indicator retained for explicit safety or endpoint logic."
    if table.startswith("gold_analysis_"):
        return "Derived analytical field retained without modifying canonical gold semantics."
    return "Approved gold delivery field."


def _variable_class(column: str) -> str:
    if column == "MUNICIPALITY_CODE":
        return "identifiers_and_keys"
    if column in {"DATE", "YEAR", "MONTH"}:
        return "calendar_and_geography"
    if column in {"TMAX_C", "TMIN_C", "TMEAN_C"}:
        return "primary_exposures"
    if column in {"RH_MIN_PCT", "RH_MEAN_PCT", "HEATWAVE_LEN_P95", "HEAT_INDEX_APPROX_C"}:
        return "humidity_and_co_exposures"
    if column.startswith("SOURCE_") or column.startswith("FONTE_") or column.startswith("IMPUTADO_") or column.endswith("_NOTE") or column.endswith("_LABEL"):
        return "lineage_and_provenance"
    if column == "QC_STATUS":
        return "quality_control_variables"
    if column in LEGACY_COMPATIBILITY_FIELDS:
        return "legacy_fields"
    if "DENOMINATOR" in column or column.startswith("OFFSET") or column.startswith("REF_") or column.startswith("POP_") or column in {
        "LIVE_BIRTHS_TOTAL",
        "LIVE_BIRTHS_TOTAL_ANNUAL",
        "TOTAL_BIRTHS_PRIMARY",
        "TOTAL_BIRTHS_SENSITIVITY_GTE22",
        "NASCIMENTOS_TOTAIS_INCLUINDO_FETAIS",
        "POP_U5",
        "UNDER5_POPULATION",
    } or "RATE_" in column or column.startswith("CRUDE_RATE"):
        return "denominators_and_offsets"
    if column in MATERNAL_PHENOTYPE_FIELDS:
        return "phenotype_flags"
    if column in SECONDARY_OR_SENSITIVITY_FIELDS or column.startswith("PRENATAL_"):
        return "secondary_or_sensitivity_outcomes"
    if column in PRIMARY_OUTCOME_FIELDS or column.startswith("NV_") or column.startswith("STILLBIRTH") or column.startswith("DEATHS_"):
        return "primary_outcomes"
    return "quality_control_variables"


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema.names)


def _parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _climate_stats(path: Path) -> dict[str, Any]:
    columns = set(_parquet_columns(path))
    if not set(CLIMATE_COLUMNS).issubset(columns):
        return {"present": False, "complete": False, "summary": "not_applicable", "missing_rows": 0}
    frame = pd.read_parquet(path, columns=list(CLIMATE_COLUMNS))
    total_rows = int(len(frame))
    complete_rows = int(frame.notna().all(axis=1).sum())
    return {
        "present": True,
        "complete": complete_rows == total_rows,
        "summary": f"{complete_rows}/{total_rows}" if total_rows else "0/0",
        "missing_rows": total_rows - complete_rows,
    }


def _all_years(start_year: int, end_year: int) -> list[int]:
    return list(range(start_year, end_year + 1))


def build_gold_table_summaries(root: Path, start_year: int, end_year: int) -> dict[str, dict[str, Any]]:
    years = _all_years(start_year, end_year)
    summaries: dict[str, dict[str, Any]] = {}
    for table in DOCUMENTED_GOLD_ITEMS:
        stage_evidence = _load_stage_evidence(root, table)
        columns_by_year: dict[int, list[str]] = {}
        row_count_by_year: dict[int, int] = {}
        climate_by_year: dict[int, dict[str, Any]] = {}
        missing_years: list[int] = []
        first_ordered_columns: list[str] = []
        for year in years:
            path = _gold_output_path(root, table, year)
            if not path.exists():
                missing_years.append(year)
                continue
            ordered_columns = _parquet_columns(path)
            if not first_ordered_columns:
                first_ordered_columns = ordered_columns
            columns_by_year[year] = ordered_columns
            row_count_by_year[year] = _parquet_row_count(path)
            climate_by_year[year] = _climate_stats(path)

        schema_consistent = bool(columns_by_year)
        if columns_by_year:
            reference = list(columns_by_year.values())[0]
            schema_consistent = all(columns == reference for columns in columns_by_year.values())

        internal_pass = all(
            str(stage_evidence.get(year, {}).get("STATUS", "FAIL")).upper() == "PASS"
            for year in years
            if year not in missing_years
        )
        external_pass = all(
            str(stage_evidence.get(year, {}).get("STATUS", "FAIL")).upper() == "PASS"
            for year in years
            if year not in missing_years
        )

        summaries[table] = {
            "table": table,
            "layer": TABLE_LAYER[table],
            "years": years,
            "missing_years": missing_years,
            "columns_by_year": columns_by_year,
            "columns": first_ordered_columns,
            "row_count_by_year": row_count_by_year,
            "schema_consistent": schema_consistent,
            "stage_evidence": stage_evidence,
            "internal_validation_pass": bool(columns_by_year) and internal_pass and not missing_years,
            "external_validation_pass": bool(columns_by_year) and external_pass and not missing_years,
            "climate_by_year": climate_by_year,
            "climate_complete": bool(climate_by_year) and all(stats.get("complete", False) for stats in climate_by_year.values()),
            "variable_count": len(first_ordered_columns),
        }
    return summaries


def build_gold_variable_dictionary(root: Path, start_year: int, end_year: int) -> dict[str, Any]:
    years = _all_years(start_year, end_year)
    canonical_descriptions = _canonical_description_index(root)
    rows: list[dict[str, Any]] = []
    table_counts: dict[str, int] = {}

    for table in DOCUMENTED_GOLD_ITEMS:
        path_candidates = [
            path
            for year in years
            if (path := _gold_output_path(root, table, year)).exists()
        ]
        if not path_candidates:
            continue
        columns = _parquet_columns(path_candidates[0])
        descriptions = _schema_description_map(root, table)
        for column in columns:
            definition = descriptions.get(column) or _inherited_definition(root, table, column, canonical_descriptions)
            rows.append(
                {
                    "variable_name": column,
                    "layer": TABLE_LAYER[table],
                    "table_name": table,
                    "variable_class": _variable_class(column),
                    "plain_english_label": _plain_label(column, definition),
                    "definition": definition,
                    "unit": _unit(column),
                    "key_role": _key_role(column),
                    "source_provenance": _source_provenance(table, column),
                    "primary_or_sensitivity": _primary_or_sensitivity(column),
                    "expected_missingness": _expected_missingness(table, column),
                    "outcome_family": _outcome_family(table, column),
                    "age_scope": _age_scope(table, column),
                    "denominator_rule": _denominator_rule(table, column),
                    "canonical_or_legacy": _canonical_or_legacy(column),
                    "notes_on_use": _notes_on_use(table, column),
                }
            )
        table_counts[table] = len(columns)

    dictionary = pd.DataFrame(rows)
    if not dictionary.empty:
        dictionary = dictionary.sort_values(["table_name", "variable_name"], kind="mergesort").reset_index(drop=True)

    required_documented = {
        "GEST_AGE_CANONICAL",
        "PRETERM_FLAG",
        "VERY_PRETERM_FLAG",
        "GEST_DAYS_LOST_BELOW_37W",
        "HEAT_INDEX_APPROX_C",
        "DENOMINATOR_ZERO_FLAG",
        "DENOMINATOR_MISSING_FLAG",
        "RATE_NOT_COMPUTABLE_FLAG",
        "OFFSET_VALID_FLAG",
        "SOURCE_DERIVED_VIEW",
        "VIEW_ROLE_NOTE",
        "WRA_AGE_SCOPE_NOTE",
        "CHILD_VIEW_AGE_LOGIC_NOTE",
        "SENSITIVITY_ONLY_FLAG",
    }
    documented_variables = set(dictionary["variable_name"].tolist()) if not dictionary.empty else set()
    missing_required = sorted(required_documented - documented_variables)
    return {
        "dictionary": dictionary,
        "table_variable_counts": table_counts,
        "required_documented": not missing_required,
        "missing_required_variables": missing_required,
    }


def _required_fields_present(summary: dict[str, Any], spec: EndpointSpec) -> bool:
    return all(field in summary["columns"] for field in spec.required_fields)


def _denominator_valid(summary: dict[str, Any], spec: EndpointSpec) -> bool:
    if spec.denominator_rule == NOT_APPLICABLE_DENOMINATOR:
        return True
    return all(field in summary["columns"] for field in spec.denominator_fields)


def _readiness_status(
    required_fields_present: bool,
    denominator_valid: bool,
    climate_complete: bool,
    internal_validation_pass: bool,
    external_validation_pass: bool,
) -> str:
    if not required_fields_present or not internal_validation_pass or not external_validation_pass:
        return "Blocked"
    if not denominator_valid or not climate_complete:
        return "Partially ready"
    return "Ready now"


def _readiness_blocker(
    required_fields_present: bool,
    denominator_valid: bool,
    climate_complete: bool,
    internal_validation_pass: bool,
    external_validation_pass: bool,
) -> str:
    blockers: list[str] = []
    if not required_fields_present:
        blockers.append("required fields missing from the approved gold artifact")
    if not denominator_valid:
        blockers.append("required denominator contract is incomplete for this endpoint")
    if not climate_complete:
        blockers.append("climate enrichment is not zero-missing across 2020-2024, although lineage is preserved")
    if not internal_validation_pass:
        blockers.append("internal validation did not fully pass across all approved years")
    if not external_validation_pass:
        blockers.append("external validation did not fully pass across all approved years")
    return " | ".join(blockers)


def _next_action_for_status(status: str, blocker: str) -> str:
    if status == "Ready now":
        return "Consume the approved gold artifact directly and do not redefine endpoint semantics downstream."
    if status == "Partially ready" and "climate enrichment is not zero-missing" in blocker:
        return "Use the approved artifact with lineage-aware climate missingness, or refresh climate backbone matching before requiring zero-missing exposures."
    if status == "Partially ready":
        return "Resolve the noted contract gap before using this endpoint as a fully model-ready analytical input."
    return "Rebuild the upstream gold stage or refresh the missing contract fields, then rerun Stage 5."


def build_gold_readiness_ledger(root: Path, start_year: int, end_year: int, table_summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    del start_year, end_year
    rows: list[dict[str, Any]] = []
    for spec in ENDPOINT_SPECS:
        summary = table_summaries[spec.gold_table]
        required_fields_present = _required_fields_present(summary, spec)
        denominator_valid = _denominator_valid(summary, spec)
        climate_complete = summary["climate_complete"] if spec.climate_required else True
        internal_validation_pass = bool(summary["internal_validation_pass"])
        external_validation_pass = bool(summary["external_validation_pass"])
        status = _readiness_status(
            required_fields_present,
            denominator_valid,
            climate_complete,
            internal_validation_pass,
            external_validation_pass,
        )
        blocker = _readiness_blocker(
            required_fields_present,
            denominator_valid,
            climate_complete,
            internal_validation_pass,
            external_validation_pass,
        )
        rows.append(
            {
                "endpoint_name": spec.endpoint_name,
                "status": status,
                "gold_table": spec.gold_table,
                "canonical_source": spec.canonical_source,
                "required_fields": "; ".join(spec.required_fields),
                "required_fields_present_yes_no": "yes" if required_fields_present else "no",
                "denominator_rule": spec.denominator_rule,
                "denominator_valid_yes_no": "yes" if denominator_valid else "no",
                "climate_enrichment_complete_yes_no": "yes" if climate_complete else "no",
                "internal_validation_pass_fail": "pass" if internal_validation_pass else "fail",
                "external_validation_pass_fail": "pass" if external_validation_pass else "fail",
                "primary_or_sensitivity": spec.primary_or_sensitivity,
                "age_scope": spec.age_scope or TABLE_AGE_SCOPE[spec.gold_table],
                "blocker": blocker,
                "next_corrective_action": _next_action_for_status(status, blocker),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["gold_table", "endpoint_name"], kind="mergesort").reset_index(drop=True)


def build_stage5_table_year_matrix(
    root: Path,
    start_year: int,
    end_year: int,
    table_summaries: dict[str, dict[str, Any]],
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    years = _all_years(start_year, end_year)
    dictionary_counts = dictionary.groupby("table_name")["variable_name"].nunique().to_dict() if not dictionary.empty else {}
    rows: list[dict[str, Any]] = []
    for table in DOCUMENTED_GOLD_ITEMS:
        summary = table_summaries[table]
        stage_evidence = summary["stage_evidence"]
        documented_count = int(dictionary_counts.get(table, 0))
        required_documented = documented_count == int(summary["variable_count"])
        for year in years:
            output_path = _gold_output_path(root, table, year)
            stage_row = stage_evidence.get(year, {})
            exists = output_path.exists()
            stage_pass = str(stage_row.get("STATUS", "FAIL")).upper() == "PASS"
            status = "PASS" if exists and stage_pass and summary["schema_consistent"] and required_documented else "FAIL"
            blockers: list[str] = []
            if not exists:
                blockers.append("gold output missing")
            if not stage_pass:
                blockers.append("upstream approved stage evidence missing or FAIL")
            if not summary["schema_consistent"]:
                blockers.append("schema varies across years")
            if not required_documented:
                blockers.append("not all observed variables were documented in the Stage 5 dictionary")
            rows.append(
                {
                    "table": table,
                    "year": year,
                    "layer": TABLE_LAYER[table],
                    "status": status,
                    "output_gold_path": str(output_path.relative_to(root)),
                    "rows": int(summary["row_count_by_year"].get(year, 0)),
                    "variable_count": int(summary["variable_count"]),
                    "documented_variable_count": documented_count,
                    "schema_consistent_with_dictionary_yes_no": "yes" if summary["schema_consistent"] else "no",
                    "required_documented_yes_no": "yes" if required_documented else "no",
                    "climate_enrichment_complete_yes_no": (
                        "yes" if summary["climate_by_year"].get(year, {}).get("complete", False) else "no"
                    ),
                    "internal_validation_pass_fail": ("pass" if stage_pass else "fail"),
                    "external_validation_pass_fail": ("pass" if stage_pass else "fail"),
                    "blocking_issue": " | ".join(blockers),
                }
            )
    return pd.DataFrame(rows).sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)


def build_stage5_mini_report(
    matrix: pd.DataFrame,
    readiness_ledger: pd.DataFrame,
    manifest_refreshed: bool,
) -> pd.DataFrame:
    readiness_counts = readiness_ledger.groupby(["gold_table", "status"]).size().unstack(fill_value=0)
    rows: list[dict[str, Any]] = []
    for row in matrix.to_dict(orient="records"):
        table = str(row["table"])
        rows.append(
            {
                "table": table,
                "year": int(row["year"]),
                "STATUS": row["status"],
                "output_gold_path": row["output_gold_path"],
                "rows": int(row["rows"]),
                "variable_count": int(row["variable_count"]),
                "documented_variable_count": int(row["documented_variable_count"]),
                "schema_consistent_with_dictionary_yes_no": row["schema_consistent_with_dictionary_yes_no"],
                "required_documented_yes_no": row["required_documented_yes_no"],
                "climate_enrichment_complete_yes_no": row["climate_enrichment_complete_yes_no"],
                "internal_validation_pass_fail": row["internal_validation_pass_fail"],
                "external_validation_pass_fail": row["external_validation_pass_fail"],
                "endpoint_ready_now_count": int(readiness_counts.loc[table, "Ready now"]) if table in readiness_counts.index and "Ready now" in readiness_counts.columns else 0,
                "endpoint_partially_ready_count": int(readiness_counts.loc[table, "Partially ready"]) if table in readiness_counts.index and "Partially ready" in readiness_counts.columns else 0,
                "endpoint_blocked_count": int(readiness_counts.loc[table, "Blocked"]) if table in readiness_counts.index and "Blocked" in readiness_counts.columns else 0,
                "manifest_refreshed_yes_no": "yes" if manifest_refreshed else "no",
                "blocker_and_corrective_action_if_FAIL": (
                    ""
                    if row["status"] == "PASS"
                    else f"{row['blocking_issue']} || corrective_action=Refresh the missing documentation or upstream gold artifact, then rerun Stage 5."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["table", "year"], kind="mergesort").reset_index(drop=True)
