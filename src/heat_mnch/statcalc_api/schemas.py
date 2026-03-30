from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TrackId = Literal["track_a", "track_b"]
ExecutionMode = Literal["reference", "live"]
RunStatus = Literal["queued", "running", "completed", "completed_with_warnings", "failed"]


class SelectorOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StratifierDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stratifier_id: str
    label: str
    description: str
    stratum_column: str
    strata_kind: str
    stratum_labels: list[str]


class AnalysisRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    track_id: TrackId
    label: str
    description: str
    outcome_family: str
    execution_behavior: Literal["sync", "async"]
    default_template_id: str
    default_offset_id: str
    templates: list[SelectorOption]
    offsets: list[SelectorOption]
    outcomes: list[SelectorOption]
    exposures: list[SelectorOption]
    lag_windows: list[SelectorOption] = Field(default_factory=list)
    stratifiers: list[str] = Field(default_factory=list)
    source_filters: list[SelectorOption] = Field(default_factory=list)
    grouping_templates: list[SelectorOption] = Field(default_factory=list)
    denominator_variants: list[SelectorOption] = Field(default_factory=list)
    gestational_source_modes: list[SelectorOption] = Field(default_factory=list)


class AnalysisRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    outcome_variable: str
    exposure_variable: str
    lag_or_window: str | None = None
    template_id: str
    stratifier_id: str | None = None
    source_filter: str = "all_sources"
    denominator_variant: str | None = None
    grouping_template: str | None = None
    gestational_source_mode: str | None = None
    offset_id: str | None = None
    adjustment_context_variables: list[str] = Field(default_factory=list)

    @field_validator("adjustment_context_variables")
    @classmethod
    def _normalize_adjustment_context_variables(cls, value: list[str]) -> list[str]:
        return [str(item) for item in value if str(item)]


class AnalysisResultRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: str
    label: str
    group_label: str | None = None
    estimate: float | None = None
    lower_ci: float | None = None
    upper_ci: float | None = None
    p_value: float | None = None
    estimate_scale: str
    model_label: str
    valid_strata_count: int = 0
    observation_count: int | None = None
    outcome_sum: float | None = None
    denominator_sum: float | None = None
    primary_row_flag: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    track_id: TrackId
    recipe_id: str
    execution_mode: ExecutionMode
    summary: dict[str, Any]
    rows: list[AnalysisResultRow]
    warnings: list[str] = Field(default_factory=list)
    forest_plot_url: str | None = None
    table_csv_url: str | None = None
    spec: AnalysisRunSpec


class AnalysisRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    track_id: TrackId
    recipe_id: str
    status: RunStatus
    execution_mode: ExecutionMode
    is_reference: bool
    spec_hash: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    run_dir: str
    result_available: bool = False
    spec: AnalysisRunSpec


class SavedPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset_id: str
    name: str
    description: str | None = None
    created_at: str
    updated_at: str
    run_spec: AnalysisRunSpec


class SavedPresetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    run_spec: AnalysisRunSpec


class SavedPresetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    run_spec: AnalysisRunSpec
