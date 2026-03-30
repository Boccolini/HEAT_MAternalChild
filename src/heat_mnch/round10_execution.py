from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path

import duckdb
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalPoisson

from heat_mnch.run_log import append_run_event, init_run_log, utc_now


ACUTE_RESULTS_DIR = Path("artifacts/analysis/step9_rebuild/results/round10")
ACUTE_INPUT_GLOB = "data/gold/gold_analysis_child_u5/**/*.parquet"
CONTEXT_PATH = "data/gold/gold_municipality_context_static/gold_municipality_context_static.parquet"

MACROREGION_CASE_SQL = """
CASE
    WHEN SUBSTR(CAST(g.MUNICIPALITY_CODE AS VARCHAR), 1, 2) IN ('11','12','13','14','15','16','17') THEN 'North'
    WHEN SUBSTR(CAST(g.MUNICIPALITY_CODE AS VARCHAR), 1, 2) IN ('21','22','23','24','25','26','27','28','29') THEN 'Northeast'
    WHEN SUBSTR(CAST(g.MUNICIPALITY_CODE AS VARCHAR), 1, 2) IN ('31','32','33','35') THEN 'Southeast'
    WHEN SUBSTR(CAST(g.MUNICIPALITY_CODE AS VARCHAR), 1, 2) IN ('41','42','43') THEN 'South'
    WHEN SUBSTR(CAST(g.MUNICIPALITY_CODE AS VARCHAR), 1, 2) IN ('50','51','52','53') THEN 'Center-West'
    ELSE 'Unknown'
END
"""


@dataclass(frozen=True)
class OutcomeSpec:
    domain_id: str
    outcome_id: str
    outcome_column: str
    primary_model_id: str
    primary_window: str
    lead_placebo_window: str
    paper_id: str
    headline_status: str
    offset_column: str | None = None
    offset_flag_column: str | None = None


OUTCOME_SPECS: tuple[OutcomeSpec, ...] = (
    OutcomeSpec(
        domain_id="acute_companion_primary",
        outcome_id="child_under5_deaths",
        outcome_column="UNDER5_DEATHS_TOTAL",
        primary_model_id="r6_conditional_poisson_u5",
        primary_window="lag0_3",
        lead_placebo_window="lead1_4",
        paper_id="acute_child_companion",
        headline_status="headline_eligible",
        offset_column="OFFSET_LOG_UNDER5_POPULATION",
        offset_flag_column="UNDER5_POPULATION_OFFSET_VALID_FLAG",
    ),
    OutcomeSpec(
        domain_id="acute_companion_supportive_mortality",
        outcome_id="child_infant_deaths",
        outcome_column="INFANT_DEATHS_TOTAL",
        primary_model_id="r6_conditional_poisson_infant",
        primary_window="lag0_3",
        lead_placebo_window="lead1_4",
        paper_id="acute_child_companion",
        headline_status="supportive_only",
        offset_column="OFFSET_LOG_LIVE_BIRTHS",
        offset_flag_column="LIVE_BIRTHS_OFFSET_VALID_FLAG",
    ),
    OutcomeSpec(
        domain_id="acute_companion_supportive_mortality",
        outcome_id="child_deaths_28_364d",
        outcome_column="DEATHS_28_364D",
        primary_model_id="r6_conditional_poisson_28_364d",
        primary_window="lag0_3",
        lead_placebo_window="lead1_4",
        paper_id="acute_child_companion",
        headline_status="supportive_only",
    ),
    OutcomeSpec(
        domain_id="acute_companion_supportive_mortality",
        outcome_id="child_deaths_1_4y",
        outcome_column="DEATHS_1_4Y",
        primary_model_id="r6_conditional_poisson_1_4y",
        primary_window="lag0_3",
        lead_placebo_window="lead1_4",
        paper_id="acute_child_companion",
        headline_status="supportive_only",
    ),
    OutcomeSpec(
        domain_id="acute_companion_core_morbidity",
        outcome_id="child_dehydration_admissions",
        outcome_column="PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL",
        primary_model_id="r6_conditional_poisson_dehydration",
        primary_window="lag0_1",
        lead_placebo_window="lead1_2",
        paper_id="acute_child_companion",
        headline_status="supportive_only",
    ),
    OutcomeSpec(
        domain_id="acute_companion_core_morbidity",
        outcome_id="child_diarrhea_gi_admissions",
        outcome_column="PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL",
        primary_model_id="r6_conditional_poisson_diarrhea_gi",
        primary_window="lag0_1",
        lead_placebo_window="lead1_2",
        paper_id="acute_child_companion",
        headline_status="supportive_only",
    ),
)

BLOCKED_DOMAIN_ROWS: tuple[dict[str, str], ...] = (
    {
        "domain_id": "flagship_preterm_primary",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "execution_surface_not_materialized",
        "blocker_reason": "Gestational-week person-period exposure history required by E002 is not yet materialized in a Step 9-native execution surface.",
        "required_execution_surface": "gestational_live_birth_weekly_exposure_person_period",
        "next_unblock_round": "reopen_round3B_or_new_execution_round",
        "notes": "Round 5 primary estimator remains frozen but not yet executable in the current Step 9-native mart layer.",
    },
    {
        "domain_id": "key_secondary_stillbirth",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "execution_surface_not_materialized",
        "blocker_reason": "Weekly fetuses-at-risk exposure history for the >=28 stillbirth backbone is not yet materialized in a Step 9-native execution surface.",
        "required_execution_surface": "stillbirth_fetuses_at_risk_weekly_exposure_surface",
        "next_unblock_round": "reopen_round3B_or_new_execution_round",
        "notes": "Round 5 stillbirth grouped hazard remains frozen but not yet executable with the required exposure history.",
    },
    {
        "domain_id": "key_secondary_neonatal",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "execution_surface_not_materialized",
        "blocker_reason": "The grouped neonatal bridge requires a dedicated municipality x birth-cohort x day-of-life execution mart that preserves the Round 5 bridge constraints.",
        "required_execution_surface": "neonatal_bridge_execution_surface",
        "next_unblock_round": "reopen_round3B_or_new_execution_round",
        "notes": "Round 5 neonatal bridge estimator remains frozen but the execution mart is not yet materialized.",
    },
    {
        "domain_id": "supportive_live_birth_phenotypes",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "manuscript_scope_barrier",
        "blocker_reason": "Supportive live-birth phenotypes are supplement-only and cannot anchor Round 10 headline synthesis.",
        "required_execution_surface": "not_applicable_in_round10",
        "next_unblock_round": "supplement_execution_after_flagship_primary",
        "notes": "Domain stays subordinate to flagship preterm synthesis.",
    },
    {
        "domain_id": "gestational_methods_sensitivity",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "manuscript_scope_barrier",
        "blocker_reason": "Methods sensitivities remain supplement-only and are blocked from main Round 10 synthesis.",
        "required_execution_surface": "not_applicable_in_round10",
        "next_unblock_round": "supplement_execution_after_flagship_primary",
        "notes": "Round 5 and Round 8 kept this domain methods-facing only.",
    },
    {
        "domain_id": "maternal_operational_secondary",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "system_facing_only",
        "blocker_reason": "Maternal operational outcomes remain supplement-only and system-facing rather than headline-synthesis eligible.",
        "required_execution_surface": "not_applicable_in_round10",
        "next_unblock_round": "separate_system_facing_follow_up",
        "notes": "Round 7 firewall remains in force.",
    },
    {
        "domain_id": "acute_companion_respiratory_sensitivity",
        "paper_id": "acute_child_companion",
        "blocker_class": "supplement_only_sensitivity",
        "blocker_reason": "Respiratory outcomes remain sensitivity-only and cannot enter Round 10 headline synthesis.",
        "required_execution_surface": "supplement_only_if_run_later",
        "next_unblock_round": "supplement_follow_up_only",
        "notes": "Round 6 and Round 8 kept this domain outside the acute headline lane.",
    },
    {
        "domain_id": "rare_thermal_internal",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "internal_only",
        "blocker_reason": "Rare thermal sentinels remain internal-only and outside Round 10 scientific synthesis.",
        "required_execution_surface": "not_applicable",
        "next_unblock_round": "internal_only",
        "notes": "Round 7 policy remains unchanged.",
    },
    {
        "domain_id": "care_process_internal",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "internal_only",
        "blocker_reason": "Care-process and proxy outcomes remain internal-only and outside Round 10 scientific synthesis.",
        "required_execution_surface": "not_applicable",
        "next_unblock_round": "internal_only",
        "notes": "Firewall against downstream or proxy outcomes remains unchanged.",
    },
    {
        "domain_id": "audit_duplicate_neonatal",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "audit_only",
        "blocker_reason": "Audit duplicate remains blocked from active synthesis and manuscript surfaces.",
        "required_execution_surface": "not_applicable",
        "next_unblock_round": "never_active",
        "notes": "Deaths 0-27 days remain provenance-only.",
    },
    {
        "domain_id": "audit_duplicate_under5",
        "paper_id": "flagship_gestational_perinatal",
        "blocker_class": "audit_only",
        "blocker_reason": "Audit duplicate remains blocked from active synthesis and manuscript surfaces.",
        "required_execution_surface": "not_applicable",
        "next_unblock_round": "never_active",
        "notes": "Total under-five deaths remain provenance-only.",
    },
)


def _build_temp_base(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE acute_round10_base AS
        WITH context_raw AS (
            SELECT
                CAST(MUNICIPALITY_CODE_IBGE7 AS BIGINT) AS MUNICIPALITY_CODE,
                CAST(IDH_EXPANDED AS DOUBLE) AS IDH_EXPANDED,
                CAST(GINI_EXPANDED AS DOUBLE) AS GINI_EXPANDED
            FROM read_parquet('{CONTEXT_PATH}')
            WHERE MUNICIPALITY_CODE_IBGE7 IS NOT NULL
        ),
        context_z AS (
            SELECT
                MUNICIPALITY_CODE,
                IDH_EXPANDED,
                GINI_EXPANDED,
                (IDH_EXPANDED - AVG(IDH_EXPANDED) OVER ())
                    / NULLIF(STDDEV_POP(IDH_EXPANDED) OVER (), 0) AS IDH_Z,
                (GINI_EXPANDED - AVG(GINI_EXPANDED) OVER ())
                    / NULLIF(STDDEV_POP(GINI_EXPANDED) OVER (), 0) AS GINI_Z
            FROM context_raw
        ),
        base AS (
            SELECT
                g.MUNICIPALITY_CODE,
                g.DATE::DATE AS DATE,
                CAST(g.YEAR AS INTEGER) AS YEAR,
                CAST(g.MONTH AS INTEGER) AS MONTH,
                CAST(EXTRACT('dow' FROM g.DATE::DATE) AS INTEGER) AS DAY_OF_WEEK,
                CONCAT(
                    CAST(g.MUNICIPALITY_CODE AS VARCHAR), '|',
                    CAST(g.YEAR AS VARCHAR), '|',
                    LPAD(CAST(g.MONTH AS VARCHAR), 2, '0'), '|',
                    CAST(EXTRACT('dow' FROM g.DATE::DATE) AS VARCHAR)
                ) AS STRATUM_ID,
                CASE
                    WHEN g.YEAR IN (2020, 2021) THEN '2020_2021'
                    ELSE '2022_2024'
                END AS SUBPERIOD,
                SUBSTR(CAST(g.MUNICIPALITY_CODE AS VARCHAR), 1, 2) AS UF_CODE,
                {MACROREGION_CASE_SQL} AS MACROREGION,
                CAST(g.TMAX_C AS DOUBLE) AS TMAX_C,
                CAST(g.HEAT_INDEX_APPROX_C AS DOUBLE) AS HEAT_INDEX_APPROX_C,
                CAST(g.HEATWAVE_LEN_P95 AS DOUBLE) AS HEATWAVE_LEN_P95,
                g.FONTE_TMAX_C_FINAL,
                g.CLIMATE_CORRECTION_PATH,
                CAST(g.UNDER5_DEATHS_TOTAL AS DOUBLE) AS UNDER5_DEATHS_TOTAL,
                CAST(g.INFANT_DEATHS_TOTAL AS DOUBLE) AS INFANT_DEATHS_TOTAL,
                CAST(g.DEATHS_28_364D AS DOUBLE) AS DEATHS_28_364D,
                CAST(g.DEATHS_1_4Y AS DOUBLE) AS DEATHS_1_4Y,
                CAST(g.PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL AS DOUBLE) AS PRINCIPAL_DIAGNOSIS_CHILD_DEHYDRATION_PRIMARY_TOTAL,
                CAST(g.PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL AS DOUBLE) AS PRINCIPAL_DIAGNOSIS_CHILD_DIARRHEA_GI_PRIMARY_TOTAL,
                CAST(g.OFFSET_LOG_UNDER5_POPULATION AS DOUBLE) AS OFFSET_LOG_UNDER5_POPULATION,
                CAST(g.OFFSET_LOG_LIVE_BIRTHS AS DOUBLE) AS OFFSET_LOG_LIVE_BIRTHS,
                CAST(g.UNDER5_POPULATION_OFFSET_VALID_FLAG AS VARCHAR) AS UNDER5_POPULATION_OFFSET_VALID_FLAG,
                CAST(g.LIVE_BIRTHS_OFFSET_VALID_FLAG AS VARCHAR) AS LIVE_BIRTHS_OFFSET_VALID_FLAG,
                c.IDH_Z,
                c.GINI_Z
            FROM read_parquet('{ACUTE_INPUT_GLOB}', union_by_name=true) AS g
            LEFT JOIN context_z AS c USING (MUNICIPALITY_CODE)
        )
        SELECT
            *,
            AVG(TMAX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN CURRENT ROW AND CURRENT ROW
            ) AS TMAX_LAG0,
            AVG(TMAX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
            ) AS TMAX_LAG0_1,
            AVG(TMAX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ) AS TMAX_LAG0_3,
            AVG(TMAX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
            ) AS TMAX_LAG0_7,
            AVG(TMAX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 1 FOLLOWING AND 2 FOLLOWING
            ) AS TMAX_LEAD1_2,
            AVG(TMAX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 1 FOLLOWING AND 4 FOLLOWING
            ) AS TMAX_LEAD1_4,
            AVG(HEAT_INDEX_APPROX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
            ) AS HEAT_INDEX_LAG0_1,
            AVG(HEAT_INDEX_APPROX_C) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ) AS HEAT_INDEX_LAG0_3,
            AVG(HEATWAVE_LEN_P95) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
            ) AS HEATWAVE_LAG0_1,
            AVG(HEATWAVE_LEN_P95) OVER (
                PARTITION BY MUNICIPALITY_CODE
                ORDER BY DATE
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ) AS HEATWAVE_LAG0_3
        FROM base
        """
    )


def _fit_conditional_poisson(
    frame: pd.DataFrame,
    outcome_column: str,
    exposure_column: str,
    interaction_column: str | None = None,
) -> dict[str, float | int | str]:
    model_frame = frame[[outcome_column, exposure_column, "STRATUM_ID"]].copy()
    if interaction_column is not None:
        model_frame[interaction_column] = frame[interaction_column]

    model_frame = model_frame.dropna()
    if model_frame.empty:
        raise ValueError("No non-missing rows available for fitting")

    informative = model_frame.groupby("STRATUM_ID", observed=True).agg(
        outcome_sum=(outcome_column, "sum"),
        exposure_var=(exposure_column, "var"),
    )
    informative = informative[
        (informative["outcome_sum"] > 0)
        & informative["exposure_var"].fillna(0).gt(0)
    ]
    if informative.empty:
        raise ValueError("No informative strata after filtering")

    filtered = model_frame[model_frame["STRATUM_ID"].isin(informative.index)].copy()
    filtered["grp_code"] = pd.Categorical(filtered["STRATUM_ID"]).codes
    predictors = [exposure_column]
    if interaction_column is not None:
        predictors.append(interaction_column)

    result = ConditionalPoisson(
        filtered[outcome_column],
        filtered[predictors],
        groups=filtered["grp_code"],
    ).fit(disp=False, maxiter=100)

    output: dict[str, float | int | str] = {
        "model_rows": int(len(filtered)),
        "informative_groups": int(filtered["grp_code"].nunique()),
        "coef_log_rr": float(result.params[exposure_column]),
        "se_log_rr": float(result.bse[exposure_column]),
        "rr_per_unit": float(exp(result.params[exposure_column])),
        "rr_ci_low": float(exp(result.params[exposure_column] - 1.96 * result.bse[exposure_column])),
        "rr_ci_high": float(exp(result.params[exposure_column] + 1.96 * result.bse[exposure_column])),
    }
    if hasattr(result, "pvalues") and exposure_column in result.pvalues.index:
        output["p_value"] = float(result.pvalues[exposure_column])
    if interaction_column is not None:
        output["interaction_coef_log_rr"] = float(result.params[interaction_column])
        output["interaction_se_log_rr"] = float(result.bse[interaction_column])
        output["interaction_rr_per_unit"] = float(exp(result.params[interaction_column]))
        output["interaction_rr_ci_low"] = float(
            exp(result.params[interaction_column] - 1.96 * result.bse[interaction_column])
        )
        output["interaction_rr_ci_high"] = float(
            exp(result.params[interaction_column] + 1.96 * result.bse[interaction_column])
        )
        if hasattr(result, "pvalues") and interaction_column in result.pvalues.index:
            output["interaction_p_value"] = float(result.pvalues[interaction_column])
    return output


def _subset_condition(variant: str, macroregion: str | None = None) -> str:
    conditions = ["1=1"]
    if variant == "source_dependence_repeat":
        conditions.append("FONTE_TMAX_C_FINAL = 'INMET'")
    elif variant == "subperiod_repeat_2020_2021":
        conditions.append("SUBPERIOD = '2020_2021'")
    elif variant == "subperiod_repeat_2022_2024":
        conditions.append("SUBPERIOD = '2022_2024'")
    elif variant == "macroregion_repeat" and macroregion is not None:
        conditions.append(f"MACROREGION = '{macroregion}'")
    return " AND ".join(conditions)


def _exposure_column(spec: OutcomeSpec, variant: str) -> str:
    if spec.primary_window == "lag0_3":
        primary = "TMAX_LAG0_3"
        heat_index = "HEAT_INDEX_LAG0_3"
        heatwave = "HEATWAVE_LAG0_3"
    elif spec.primary_window == "lag0_1":
        primary = "TMAX_LAG0_1"
        heat_index = "HEAT_INDEX_LAG0_1"
        heatwave = "HEATWAVE_LAG0_1"
    else:
        raise ValueError(f"Unsupported primary window: {spec.primary_window}")

    if variant in {"primary", "source_dependence_repeat", "subperiod_repeat_2020_2021", "subperiod_repeat_2022_2024", "macroregion_repeat", "idh_interaction_repeat"}:
        return primary
    if variant == "heat_index_repeat":
        return heat_index
    if variant == "heatwave_duration_repeat":
        return heatwave
    if variant == "lead_day_placebo_repeat":
        if spec.lead_placebo_window == "lead1_4":
            return "TMAX_LEAD1_4"
        if spec.lead_placebo_window == "lead1_2":
            return "TMAX_LEAD1_2"
        raise ValueError(f"Unsupported lead placebo window: {spec.lead_placebo_window}")
    raise ValueError(f"Unsupported variant: {variant}")


def _event_statistics(con: duckdb.DuckDBPyConnection, spec: OutcomeSpec, variant: str, macroregion: str | None = None) -> dict[str, float | int]:
    subset = _subset_condition(variant, macroregion)
    query = f"""
        SELECT
            COALESCE(SUM({spec.outcome_column}), 0) AS total_events,
            COALESCE(SUM(CASE WHEN {spec.outcome_column} > 0 THEN 1 ELSE 0 END), 0) AS event_days,
            COALESCE(COUNT(DISTINCT CASE WHEN {spec.outcome_column} > 0 THEN MUNICIPALITY_CODE END), 0) AS municipalities_with_event_day
        FROM acute_round10_base
        WHERE {subset}
    """
    row = con.execute(query).fetchone()
    return {
        "total_events": float(row[0] or 0),
        "event_days": int(row[1] or 0),
        "municipalities_with_event_day": int(row[2] or 0),
    }


def _load_frame(
    con: duckdb.DuckDBPyConnection,
    spec: OutcomeSpec,
    variant: str,
    macroregion: str | None = None,
    include_interaction: bool = False,
) -> pd.DataFrame:
    subset = _subset_condition(variant, macroregion)
    exposure_column = _exposure_column(spec, variant)
    columns = [
        "STRATUM_ID",
        spec.outcome_column,
        exposure_column,
    ]
    if include_interaction:
        columns.extend(["IDH_Z", f"{exposure_column} * IDH_Z AS INTERACTION_TERM"])

    query = f"""
        SELECT
            {", ".join(columns)}
        FROM acute_round10_base
        WHERE {subset}
          AND {exposure_column} IS NOT NULL
    """
    frame = con.execute(query).fetchdf()
    if include_interaction and not frame.empty:
        frame["INTERACTION_TERM"] = frame[exposure_column] * frame["IDH_Z"]
    return frame


def _should_suppress(domain_id: str, variant: str, stats: dict[str, float | int]) -> tuple[bool, str]:
    if domain_id not in {
        "acute_companion_primary",
        "acute_companion_supportive_mortality",
        "acute_companion_core_morbidity",
    }:
        return False, ""

    if variant in {"macroregion_repeat", "subperiod_repeat_2020_2021", "subperiod_repeat_2022_2024"}:
        if stats["total_events"] < 100 or stats["event_days"] < 30:
            return True, "suppressed_by_round8_split_threshold"
    if variant == "idh_interaction_repeat":
        if stats["total_events"] < 500 or stats["municipalities_with_event_day"] < 100:
            return True, "suppressed_by_round8_interaction_threshold"
    return False, ""


def _variant_rows(con: duckdb.DuckDBPyConnection, spec: OutcomeSpec) -> list[dict[str, object]]:
    variants: list[tuple[str, str | None, bool]] = [
        ("primary", None, False),
        ("heat_index_repeat", None, False),
        ("source_dependence_repeat", None, False),
        ("subperiod_repeat_2020_2021", None, False),
        ("subperiod_repeat_2022_2024", None, False),
        ("lead_day_placebo_repeat", None, False),
        ("idh_interaction_repeat", None, True),
    ]
    for macroregion in ("North", "Northeast", "Southeast", "South", "Center-West"):
        variants.append(("macroregion_repeat", macroregion, False))

    output_rows: list[dict[str, object]] = []
    for variant, macroregion, include_interaction in variants:
        stats = _event_statistics(con, spec, variant, macroregion)
        suppress, suppression_reason = _should_suppress(spec.domain_id, variant, stats)
        row: dict[str, object] = {
            "domain_id": spec.domain_id,
            "paper_id": spec.paper_id,
            "outcome_id": spec.outcome_id,
            "outcome_column": spec.outcome_column,
            "primary_model_id": spec.primary_model_id,
            "variant_id": variant if macroregion is None else f"{variant}_{macroregion.lower().replace('-', '_')}",
            "variant_class": variant,
            "subset_label": macroregion or "full",
            "exposure_column": _exposure_column(spec, variant),
            "headline_status": spec.headline_status,
            "total_events": stats["total_events"],
            "event_days": stats["event_days"],
            "municipalities_with_event_day": stats["municipalities_with_event_day"],
            "suppression_status": "suppressed" if suppress else "not_suppressed",
            "suppression_reason": suppression_reason or "",
            "execution_status": "suppressed_before_fit" if suppress else "fit_pending",
        }
        if suppress:
            output_rows.append(row)
            continue

        frame = _load_frame(con, spec, variant, macroregion, include_interaction)
        try:
            fit = _fit_conditional_poisson(
                frame=frame,
                outcome_column=spec.outcome_column,
                exposure_column=_exposure_column(spec, variant),
                interaction_column="INTERACTION_TERM" if include_interaction else None,
            )
            row.update(fit)
            row["execution_status"] = "fit_complete"
        except Exception as exc:  # pragma: no cover - runtime guard
            row["execution_status"] = "fit_failed"
            row["suppression_status"] = "failed"
            row["suppression_reason"] = str(exc)
        output_rows.append(row)
    return output_rows


def _primary_result(direction_frame: pd.DataFrame) -> str:
    if direction_frame.empty:
        return "no_primary_result"
    rr = float(direction_frame.iloc[0]["rr_per_unit"])
    lo = float(direction_frame.iloc[0]["rr_ci_low"])
    hi = float(direction_frame.iloc[0]["rr_ci_high"])
    if lo > 1:
        return "positive_with_95ci_above_null"
    if hi < 1:
        return "inverse_with_95ci_below_null"
    if rr > 1:
        return "positive_but_imprecise"
    if rr < 1:
        return "inverse_but_imprecise"
    return "near_null"


def _required_repeat_assessment(result_frame: pd.DataFrame) -> tuple[str, str]:
    required = result_frame[
        result_frame["variant_class"].isin(
            {
                "heat_index_repeat",
                "source_dependence_repeat",
                "subperiod_repeat_2020_2021",
                "subperiod_repeat_2022_2024",
                "lead_day_placebo_repeat",
                "idh_interaction_repeat",
                "macroregion_repeat",
            }
        )
    ].copy()
    if required.empty:
        return "no_required_repeats_found", "No required repeat rows were found."

    failed = required[required["execution_status"] != "fit_complete"]
    if not failed.empty:
        return "required_repeat_failures_present", "One or more required repeats failed or were suppressed."

    primary_rr = float(
        result_frame.loc[result_frame["variant_class"] == "primary", "rr_per_unit"].iloc[0]
    )
    same_direction = 0
    opposite_direction = 0
    for rr in required["rr_per_unit"]:
        if rr is None:
            continue
        if primary_rr >= 1 and rr >= 1:
            same_direction += 1
        elif primary_rr <= 1 and rr <= 1:
            same_direction += 1
        else:
            opposite_direction += 1

    if opposite_direction == 0:
        return "directionally_coherent_required_repeats", "Required repeats preserved the primary direction."
    return "mixed_direction_required_repeats", "At least one required repeat reversed direction relative to the primary model."


def _domain_summaries(results: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for domain_id, frame in results.groupby("domain_id", sort=False):
        primary_rows = frame[frame["variant_class"] == "primary"].copy()
        primary_statuses = [
            _primary_result(primary_rows[primary_rows["outcome_id"] == outcome_id])
            for outcome_id in primary_rows["outcome_id"].unique()
        ]
        repeat_status, repeat_note = _required_repeat_assessment(frame)
        summaries.append(
            {
                "domain_id": domain_id,
                "paper_id": frame["paper_id"].iloc[0],
                "outcomes_executed": ";".join(primary_rows["outcome_id"].tolist()),
                "primary_result_pattern": ";".join(primary_statuses),
                "required_repeat_status": repeat_status,
                "required_repeat_note": repeat_note,
                "fit_complete_models": int((frame["execution_status"] == "fit_complete").sum()),
                "suppressed_or_failed_models": int((frame["execution_status"] != "fit_complete").sum()),
                "domain_execution_status": (
                    "executed_with_required_repeats"
                    if repeat_status != "required_repeat_failures_present"
                    else "executed_with_repeat_failures"
                ),
            }
        )
    return pd.DataFrame(summaries)


def run_round10_execution(root: Path) -> dict[str, Path]:
    results_dir = root / ACUTE_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    run_log = init_run_log(root, "round10_step9_execution")
    append_run_event(
        run_log,
        phase="round10",
        dataset="gold_analysis_child_u5",
        year=None,
        command="build acute_round10_base",
        decision="execute acute companion domains with Round 6 and Round 8 governance",
        status="started",
    )

    con = duckdb.connect()
    _build_temp_base(con)

    result_rows: list[dict[str, object]] = []
    for spec in OUTCOME_SPECS:
        append_run_event(
            run_log,
            phase="round10",
            dataset=spec.outcome_id,
            year=None,
            command="conditional_poisson_and_required_repeats",
            decision="execute primary and required Round 8 repeat bundle",
            status="started",
        )
        result_rows.extend(_variant_rows(con, spec))
        append_run_event(
            run_log,
            phase="round10",
            dataset=spec.outcome_id,
            year=None,
            command="conditional_poisson_and_required_repeats",
            decision="execute primary and required Round 8 repeat bundle",
            status="completed",
        )

    result_frame = pd.DataFrame(result_rows)
    domain_summary = _domain_summaries(result_frame)

    result_path = results_dir / "round10_acute_outcome_model_results.csv"
    domain_path = results_dir / "round10_acute_domain_summary.csv"
    result_frame.to_csv(result_path, index=False)
    domain_summary.to_csv(domain_path, index=False)

    append_run_event(
        run_log,
        phase="round10",
        dataset="acute_companion_domains",
        year=None,
        command="write round10 acute result artifacts",
        decision="persist outcome-level and domain-level execution outputs",
        status="completed",
    )

    return {
        "run_log": run_log,
        "result_path": result_path,
        "domain_path": domain_path,
    }


def materialize_round10_synthesis(root: Path) -> dict[str, Path]:
    results_dir = root / ACUTE_RESULTS_DIR
    result_path = results_dir / "round10_acute_outcome_model_results.csv"
    domain_path = results_dir / "round10_acute_domain_summary.csv"
    registry_path = root / "artifacts/analysis/step9_rebuild/registries/round10_domain_synthesis_registry.csv"
    execution_ledger_path = root / "artifacts/analysis/step9_rebuild/registries/round10_execution_ledger.csv"

    result_frame = pd.read_csv(result_path)
    domain_summary = pd.read_csv(domain_path)
    registry = pd.read_csv(registry_path)
    execution_ledger = pd.read_csv(execution_ledger_path)

    synthesized = registry.merge(domain_summary, how="left", on=["domain_id", "paper_id"])
    synthesized = synthesized.merge(
        execution_ledger[
            [
                "domain_id",
                "primary_estimator_id",
                "execution_status",
                "robustness_completion_status",
                "mnch_translation_gate_status",
                "warning_translation_gate_status",
                "synthesis_ready_status",
            ]
        ],
        how="left",
        on="domain_id",
    )

    synthesis_statement = []
    for _, row in synthesized.iterrows():
        if pd.notna(row.get("primary_result_pattern")):
            statement = (
                f"Round 10 executed {row['outcomes_executed']} with primary pattern "
                f"{row['primary_result_pattern']} and required repeat status "
                f"{row['required_repeat_status']}."
            )
        elif row["domain_id"] in {
            "flagship_preterm_primary",
            "key_secondary_stillbirth",
            "key_secondary_neonatal",
        }:
            statement = "Domain remains blocked pending Step 9-native execution-surface materialization."
        else:
            statement = "Domain remains blocked from main Round 10 synthesis by earlier architecture or audit rules."
        synthesis_statement.append(statement)
    synthesized["round10_synthesis_statement"] = synthesis_statement

    execution_linked_path = results_dir / "round10_execution_linked_synthesis_table.csv"
    synthesized.to_csv(execution_linked_path, index=False)

    blocked_path = results_dir / "round10_blocked_domain_register.csv"
    pd.DataFrame(BLOCKED_DOMAIN_ROWS).to_csv(blocked_path, index=False)

    return {
        "execution_linked_path": execution_linked_path,
        "blocked_path": blocked_path,
        "result_path": result_path,
        "domain_path": domain_path,
    }


def main() -> None:
    root = Path.cwd()
    outputs = run_round10_execution(root)
    print(outputs["result_path"])
    print(outputs["domain_path"])
    print(outputs["run_log"])


if __name__ == "__main__":
    main()
