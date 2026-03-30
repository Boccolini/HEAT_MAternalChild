from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis_track_a.modeling import (
    ACUTE_OUTCOMES,
    ACUTE_TEMPERATURE_LAG_SUFFIXES,
    BASE_EXPOSURES,
    DELIVERY_WINDOW_SUFFIXES,
    MATERNAL_OUTCOMES,
    PERINATAL_OUTCOMES,
)
from ..analysis_track_a.sensitivity import MODIFIER_SPECS
from .schemas import AnalysisRecipe, AnalysisRunSpec, SelectorOption, StratifierDefinition


TRACK_A_OUTCOME_SPECS = ACUTE_OUTCOMES + PERINATAL_OUTCOMES + MATERNAL_OUTCOMES
TRACK_A_OUTCOME_LOOKUP = {row["outcome_variable"]: row for row in TRACK_A_OUTCOME_SPECS}

TRACK_A_OFFSET_LOOKUP = {
    "daily_live_birth_count": SelectorOption(
        id="daily_live_birth_count_offset_proxy",
        label="Daily live-birth offset proxy",
        description="Automatic offset proxy derived from daily live births.",
    ),
    "annual_live_birth_count": SelectorOption(
        id="log_annual_live_birth_offset",
        label="Log annual live-birth offset",
        description="Automatic offset derived from annual live births.",
    ),
    "under_five_population_count": SelectorOption(
        id="log_under_five_population_offset",
        label="Log under-five population offset",
        description="Automatic offset derived from under-five population.",
    ),
    "total_birth_count_primary_definition": SelectorOption(
        id="log_total_birth_count_primary_offset",
        label="Log total-birth offset (primary)",
        description="Automatic offset derived from total births under the primary definition.",
    ),
    "annual_total_birth_count_primary_definition": SelectorOption(
        id="log_annual_total_birth_count_primary_offset",
        label="Log annual total-birth offset (primary)",
        description="Automatic annual offset derived from total births under the primary definition.",
    ),
    "total_birth_count_sensitivity_gte22_definition": SelectorOption(
        id="log_total_birth_count_sensitivity_gte22_offset",
        label="Log total-birth offset (sensitivity >=22 weeks)",
        description="Automatic offset derived from the sensitivity total-birth denominator.",
    ),
    "annual_total_birth_count_sensitivity_gte22_definition": SelectorOption(
        id="log_annual_total_birth_count_sensitivity_gte22_offset",
        label="Log annual total-birth offset (sensitivity >=22 weeks)",
        description="Automatic annual offset derived from the sensitivity total-birth denominator.",
    ),
    "women_age_15_49_population_count": SelectorOption(
        id="log_women_age_15_49_population_offset",
        label="Log women aged 15-49 population offset",
        description="Automatic offset derived from the women aged 15 to 49 denominator.",
    ),
    "women_age_15_45_sensitivity_population_count": SelectorOption(
        id="log_women_age_15_45_sensitivity_population_offset",
        label="Log women aged 15-45 sensitivity population offset",
        description="Automatic offset derived from the women aged 15 to 45 sensitivity denominator.",
    ),
}


TRACK_B_FALLBACK_RECIPES: dict[str, dict[str, Any]] = {
    "track_b_preterm_fe_proxy": {
        "label": "Event-level preterm grouped FE model",
        "description": "Escalation-round preterm grouped fixed-effects models.",
        "outcome_family": "track_b_preterm",
        "execution_behavior": "async",
        "default_offset_id": "birth_count_proxy_offset",
        "offset_label": "Birth count proxy offset",
        "outcomes": [
            SelectorOption(id="preterm_birth_flag", label="Preterm birth"),
            SelectorOption(id="very_preterm_birth_flag", label="Very preterm birth"),
            SelectorOption(id="gestational_days_lost_below_37_weeks", label="Gestational days lost below 37 weeks"),
        ],
        "exposures": [
            SelectorOption(
                id="third_trimester_mean_tmax_c",
                label="Third-trimester mean maximum temperature",
                metadata={"lag_or_window": "third_trimester"},
            ),
            SelectorOption(
                id="days_0_27_before_event_mean_tmax_c",
                label="Mean maximum temperature during the 28 days before delivery",
                metadata={"lag_or_window": "days_0_27_before_event"},
            ),
            SelectorOption(
                id="whole_pregnancy_mean_tmax_c",
                label="Whole-pregnancy mean maximum temperature",
                metadata={"lag_or_window": "whole_pregnancy"},
            ),
            SelectorOption(
                id="third_trimester_mean_tmax_x_rh_mean_pct",
                label="Third-trimester temperature by humidity interaction proxy",
                metadata={"lag_or_window": "third_trimester"},
            ),
            SelectorOption(
                id="third_trimester_heatwave_days",
                label="Third-trimester heatwave days",
                metadata={"lag_or_window": "third_trimester"},
            ),
            SelectorOption(
                id="third_trimester_mean_heat_index_c",
                label="Third-trimester mean heat index",
                metadata={"lag_or_window": "third_trimester"},
            ),
        ],
        "gestational_source_modes": [
            SelectorOption(id="all_sources", label="All gestational-age sources"),
            SelectorOption(id="exact_only", label="Exact gestational-age measurements only"),
        ],
    },
    "track_b_stillbirth_fe_proxy": {
        "label": "Event-level stillbirth grouped FE model",
        "description": "Escalation-round stillbirth grouped fixed-effects models.",
        "outcome_family": "track_b_stillbirth",
        "execution_behavior": "async",
        "default_offset_id": "total_birth_proxy_offset",
        "offset_label": "Total-birth proxy offset",
        "outcomes": [
            SelectorOption(id="stillbirth_primary_gte28_flag", label="Stillbirth at or beyond 28 weeks"),
            SelectorOption(id="stillbirth_sensitivity_gte22_flag", label="Stillbirth at or beyond 22 weeks"),
        ],
        "exposures": [
            SelectorOption(
                id="days_0_6_before_event_mean_tmax_c",
                label="Mean maximum temperature during the seven days before the event",
                metadata={"lag_or_window": "days_0_6_before_event"},
            ),
            SelectorOption(
                id="days_0_13_before_event_mean_tmax_c",
                label="Mean maximum temperature during the 14 days before the event",
                metadata={"lag_or_window": "days_0_13_before_event"},
            ),
            SelectorOption(
                id="days_0_27_before_event_mean_tmax_c",
                label="Mean maximum temperature during the 28 days before the event",
                metadata={"lag_or_window": "days_0_27_before_event"},
            ),
            SelectorOption(
                id="third_trimester_mean_tmax_c",
                label="Third-trimester mean maximum temperature",
                metadata={"lag_or_window": "third_trimester"},
            ),
            SelectorOption(
                id="days_0_6_before_event_mean_heat_index_c",
                label="Mean heat index during the seven days before the event",
                metadata={"lag_or_window": "days_0_6_before_event"},
            ),
        ],
        "gestational_source_modes": [
            SelectorOption(id="all_sources", label="All gestational-age sources"),
        ],
    },
    "track_b_neonatal_fe_proxy": {
        "label": "Event-level neonatal grouped FE model",
        "description": "Escalation-round neonatal grouped fixed-effects models.",
        "outcome_family": "track_b_neonatal",
        "execution_behavior": "async",
        "default_offset_id": "log_annual_live_birth_offset",
        "offset_label": "Log annual live-birth offset",
        "outcomes": [
            SelectorOption(id="neonatal_death_count", label="Neonatal mortality"),
        ],
        "exposures": [
            SelectorOption(id="lag0_tmax_c", label="Lag 0 maximum temperature", metadata={"lag_or_window": "lag0_tmax_c"}),
            SelectorOption(id="lag0_1_mean_tmax_c", label="Lag 0-1 mean maximum temperature", metadata={"lag_or_window": "lag0_1_mean_tmax_c"}),
            SelectorOption(id="lag0_3_mean_tmax_c", label="Lag 0-3 mean maximum temperature", metadata={"lag_or_window": "lag0_3_mean_tmax_c"}),
            SelectorOption(id="lag0_7_mean_tmax_c", label="Lag 0-7 mean maximum temperature", metadata={"lag_or_window": "lag0_7_mean_tmax_c"}),
            SelectorOption(id="lag0_14_mean_tmax_c", label="Lag 0-14 mean maximum temperature", metadata={"lag_or_window": "lag0_14_mean_tmax_c"}),
        ],
        "gestational_source_modes": [],
    },
}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


class StatCalcRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.stratifiers = self._build_stratifiers()
        self.stratifier_map = {item.stratifier_id: item for item in self.stratifiers}
        self.recipes = self._build_recipes()
        self.recipe_map = {item.recipe_id: item for item in self.recipes}

    def _build_stratifiers(self) -> list[StratifierDefinition]:
        definitions: list[StratifierDefinition] = []
        for item in MODIFIER_SPECS:
            labels = [
                label
                for label in [item["low_label"], item["middle_label"], item["high_label"]]
                if str(label)
            ]
            definitions.append(
                StratifierDefinition(
                    stratifier_id=item["modifier_name"],
                    label=item["modifier_label"],
                    description=f"Contextual heterogeneity stratifier based on {item['modifier_label'].lower()}.",
                    stratum_column=item["stratum_column"],
                    strata_kind=item["strata_kind"],
                    stratum_labels=labels,
                )
            )
        return definitions

    def _build_recipes(self) -> list[AnalysisRecipe]:
        recipes: list[AnalysisRecipe] = []
        recipes.append(self._build_track_a_recipe("track_a_acute_fe_proxy", "acute_daily", ACUTE_OUTCOMES, ACUTE_TEMPERATURE_LAG_SUFFIXES))
        recipes.append(
            self._build_track_a_recipe(
                "track_a_perinatal_fe_proxy",
                "perinatal_delivery_window_aggregate",
                PERINATAL_OUTCOMES,
                DELIVERY_WINDOW_SUFFIXES,
            )
        )
        recipes.append(
            self._build_track_a_recipe(
                "track_a_maternal_fe_proxy",
                "maternal_operational",
                MATERNAL_OUTCOMES,
                DELIVERY_WINDOW_SUFFIXES,
            )
        )
        recipes.extend(self._build_track_b_recipes())
        return recipes

    def _build_track_a_recipe(
        self,
        recipe_id: str,
        outcome_family: str,
        outcome_specs: list[dict[str, str]],
        lag_windows: list[str],
    ) -> AnalysisRecipe:
        outcomes = [
            SelectorOption(
                id=item["outcome_variable"],
                label=item["outcome_label"],
                metadata={"offset_id": TRACK_A_OFFSET_LOOKUP[str(item["denominator_column"])].id},
            )
            for item in outcome_specs
        ]
        offset_options = list(
            {
                TRACK_A_OFFSET_LOOKUP[str(item["denominator_column"])].id: TRACK_A_OFFSET_LOOKUP[str(item["denominator_column"])]
                for item in outcome_specs
            }.values()
        )
        default_offset = TRACK_A_OFFSET_LOOKUP[str(outcome_specs[0]["denominator_column"])]
        lag_options = [
            SelectorOption(id=lag_window, label=lag_window.replace("_", " "))
            for lag_window in lag_windows
        ]
        exposures = [
            SelectorOption(
                id=key,
                label=value["label"],
                description=value["family_name"],
            )
            for key, value in BASE_EXPOSURES.items()
        ]
        return AnalysisRecipe(
            recipe_id=recipe_id,
            track_id="track_a",
            label=recipe_id.replace("_", " ").title(),
            description="Aggregated current-cycle fixed-effects recipe built from the validated analytical marts.",
            outcome_family=outcome_family,
            execution_behavior="sync",
            default_template_id="track_a_benchmark_proxy",
            default_offset_id=default_offset.id,
            templates=[
                SelectorOption(
                    id="track_a_benchmark_proxy",
                    label="Benchmark FE proxy",
                    description="Validated aggregated current-cycle fixed-effects template.",
                )
            ],
            offsets=offset_options,
            outcomes=outcomes,
            exposures=exposures,
            lag_windows=lag_options,
            stratifiers=[item.stratifier_id for item in self.stratifiers],
            source_filters=[
                SelectorOption(id="all_sources", label="All climate rows"),
                SelectorOption(id="no_fallback", label="No fallback-imputed climate rows"),
            ],
            grouping_templates=[
                SelectorOption(id="track_a_municipality_year", label="Municipality-year FE proxy"),
            ],
            denominator_variants=[],
            gestational_source_modes=[],
        )

    def _build_track_b_recipes(self) -> list[AnalysisRecipe]:
        preterm = _read_csv_if_exists(self.root / "artifacts/track_b/models/preterm_main_models.csv")
        severity = _read_csv_if_exists(self.root / "artifacts/track_b/models/gest_days_lost_models.csv")
        stillbirth_main = _read_csv_if_exists(self.root / "artifacts/track_b/models/stillbirth_main_models.csv")
        stillbirth_sensitivity = _read_csv_if_exists(self.root / "artifacts/track_b/models/stillbirth_sensitivity_models.csv")
        neonatal = _read_csv_if_exists(self.root / "artifacts/track_b/models/neonatal_main_models.csv")

        recipe_frames = {
            "track_b_preterm_fe_proxy": pd.concat([preterm, severity], ignore_index=True) if not preterm.empty or not severity.empty else pd.DataFrame(),
            "track_b_stillbirth_fe_proxy": pd.concat([stillbirth_main, stillbirth_sensitivity], ignore_index=True) if not stillbirth_main.empty or not stillbirth_sensitivity.empty else pd.DataFrame(),
            "track_b_neonatal_fe_proxy": neonatal,
        }
        recipes: list[AnalysisRecipe] = []
        for recipe_id, fallback in TRACK_B_FALLBACK_RECIPES.items():
            frame = recipe_frames[recipe_id]
            outcomes = fallback["outcomes"]
            exposures = fallback["exposures"]
            if not frame.empty:
                if "outcome_variable" in frame.columns and "outcome_label" in frame.columns:
                    outcomes = [
                        SelectorOption(id=str(row["outcome_variable"]), label=str(row["outcome_label"]))
                        for _, row in frame.loc[:, ["outcome_variable", "outcome_label"]].drop_duplicates().iterrows()
                    ]
                if "exposure_variable" in frame.columns and "exposure_label" in frame.columns:
                    exposures = [
                        SelectorOption(
                            id=str(row["exposure_variable"]),
                            label=str(row["exposure_label"]),
                            metadata={"lag_or_window": str(row.get("lag_or_window") or row.get("exposure_variable"))},
                        )
                        for _, row in frame.loc[:, ["exposure_variable", "exposure_label"]].drop_duplicates().iterrows()
                    ]
            offset = SelectorOption(
                id=fallback["default_offset_id"],
                label=fallback["offset_label"],
                description="Automatic offset applied by the recipe.",
            )
            recipes.append(
                AnalysisRecipe(
                    recipe_id=recipe_id,
                    track_id="track_b",
                    label=fallback["label"],
                    description=fallback["description"],
                    outcome_family=fallback["outcome_family"],
                    execution_behavior=fallback["execution_behavior"],
                    default_template_id="track_b_flagship_grouped_fe",
                    default_offset_id=fallback["default_offset_id"],
                    templates=[
                        SelectorOption(
                            id="track_b_flagship_grouped_fe",
                            label="Flagship grouped FE proxy",
                            description="Grouped fixed-effects model aligned to the event-level escalation analysis.",
                        )
                    ],
                    offsets=[offset],
                    outcomes=outcomes,
                    exposures=exposures,
                    lag_windows=[],
                    stratifiers=[item.stratifier_id for item in self.stratifiers],
                    source_filters=[
                        SelectorOption(id="all_sources", label="All observed rows"),
                        SelectorOption(id="no_fallback", label="No fallback-imputed climate rows"),
                    ],
                    grouping_templates=[
                        SelectorOption(id="track_b_flagship_default", label="Flagship fallback grouping"),
                    ],
                    denominator_variants=[],
                    gestational_source_modes=fallback["gestational_source_modes"],
                )
            )
        return recipes

    def get_recipes(self) -> list[AnalysisRecipe]:
        return self.recipes

    def get_stratifiers(self) -> list[StratifierDefinition]:
        return self.stratifiers

    def validate_spec(self, spec: AnalysisRunSpec) -> tuple[AnalysisRecipe, AnalysisRunSpec]:
        recipe = self.recipe_map.get(spec.recipe_id)
        if recipe is None:
            raise ValueError(f"Unknown recipe {spec.recipe_id!r}.")
        if spec.adjustment_context_variables:
            raise ValueError("Context variables are restricted to stratification and heterogeneity, not adjustment.")

        if spec.template_id not in {item.id for item in recipe.templates}:
            raise ValueError(f"Template {spec.template_id!r} is not valid for recipe {spec.recipe_id!r}.")
        if spec.outcome_variable not in {item.id for item in recipe.outcomes}:
            raise ValueError(f"Outcome {spec.outcome_variable!r} is not valid for recipe {spec.recipe_id!r}.")
        if spec.exposure_variable not in {item.id for item in recipe.exposures}:
            raise ValueError(f"Exposure {spec.exposure_variable!r} is not valid for recipe {spec.recipe_id!r}.")
        if spec.stratifier_id and spec.stratifier_id not in set(recipe.stratifiers):
            raise ValueError(f"Stratifier {spec.stratifier_id!r} is not allowed for recipe {spec.recipe_id!r}.")
        if spec.source_filter not in {item.id for item in recipe.source_filters}:
            raise ValueError(f"Source filter {spec.source_filter!r} is not allowed for recipe {spec.recipe_id!r}.")
        if spec.grouping_template and spec.grouping_template not in {item.id for item in recipe.grouping_templates}:
            raise ValueError(f"Grouping template {spec.grouping_template!r} is not allowed for recipe {spec.recipe_id!r}.")
        if spec.gestational_source_mode and spec.gestational_source_mode not in {item.id for item in recipe.gestational_source_modes}:
            raise ValueError(
                f"Gestational source mode {spec.gestational_source_mode!r} is not allowed for recipe {spec.recipe_id!r}."
            )
        if spec.offset_id and spec.offset_id != recipe.default_offset_id:
            if recipe.track_id != "track_a":
                raise ValueError(
                    f"Offset {spec.offset_id!r} is not compatible with recipe {spec.recipe_id!r}; offsets are automatic."
                )

        lag_or_window = spec.lag_or_window
        exposure_option = next(item for item in recipe.exposures if item.id == spec.exposure_variable)
        outcome_option = next(item for item in recipe.outcomes if item.id == spec.outcome_variable)
        resolved_offset_id = str(outcome_option.metadata.get("offset_id") or recipe.default_offset_id)
        if spec.offset_id and spec.offset_id != resolved_offset_id:
            raise ValueError(
                f"Offset {spec.offset_id!r} is not compatible with outcome {spec.outcome_variable!r}; offsets are automatic."
            )
        if lag_or_window is None and exposure_option.metadata.get("lag_or_window"):
            lag_or_window = str(exposure_option.metadata["lag_or_window"])
        if recipe.lag_windows:
            valid_lags = {item.id for item in recipe.lag_windows}
            if lag_or_window not in valid_lags:
                raise ValueError(f"Lag/window {lag_or_window!r} is not valid for recipe {spec.recipe_id!r}.")

        return recipe, spec.model_copy(
            update={
                "lag_or_window": lag_or_window,
                "offset_id": resolved_offset_id,
                "grouping_template": spec.grouping_template or recipe.grouping_templates[0].id if recipe.grouping_templates else None,
                "gestational_source_mode": spec.gestational_source_mode
                or (recipe.gestational_source_modes[0].id if recipe.gestational_source_modes else None),
            }
        )

    def track_a_outcome_spec(self, outcome_variable: str) -> dict[str, str]:
        if outcome_variable not in TRACK_A_OUTCOME_LOOKUP:
            raise KeyError(outcome_variable)
        return TRACK_A_OUTCOME_LOOKUP[outcome_variable]

    def stratifier_definition(self, stratifier_id: str) -> StratifierDefinition:
        if stratifier_id not in self.stratifier_map:
            raise KeyError(stratifier_id)
        return self.stratifier_map[stratifier_id]
