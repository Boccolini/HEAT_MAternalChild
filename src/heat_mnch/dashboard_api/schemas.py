from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Granularity = Literal["week", "month", "season", "year"]
GeoLevel = Literal["national", "state", "municipality", "region"]
ValueMode = Literal["count", "rate"]
DefinitionType = Literal["primary", "sensitivity", "canonical"]
CompareMode = Literal["none", "state_vs_national", "municipality_vs_state_vs_national"]


class DashboardFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str | None = None
    metric_id: str | None = None
    years: list[int] = Field(default_factory=list)
    granularity: Granularity = "month"
    geo_level: GeoLevel = "national"
    geo_codes: list[str] = Field(default_factory=list)
    value_mode: ValueMode = "count"
    definition_type: DefinitionType | None = None
    context_variable: str | None = None
    context_bins: list[str] = Field(default_factory=list)
    compare_mode: CompareMode = "none"

    @field_validator("years")
    @classmethod
    def _sort_years(cls, value: list[int]) -> list[int]:
        return sorted({int(item) for item in value})

    @field_validator("geo_codes", "context_bins")
    @classmethod
    def _normalize_str_lists(cls, value: list[str]) -> list[str]:
        return [str(item) for item in value if str(item)]


class MetricCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str
    label: str
    value: float | None
    unit: str
    period_label: str | None = None
    delta_vs_prior_period: float | None
    quality_note: str | None


class SeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: str
    y: float | None
    denominator: float | None
    quality_flag_any: bool
    tooltip_meta: dict[str, Any]


class MapDatum(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geo_code: str
    label: str
    value: float | None
    denominator: float | None
    quality_flag_any: bool


class RankingRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geo_code: str
    label: str
    value: float | None
    rank: int
    denominator: float | None
    quality_flag_any: bool


class StratificationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stratum: str
    value: float | None
    denominator: float | None
    n_geographies: int


class MunicipalityProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    municipality_code: str
    label: str
    uf: str
    context: dict[str, Any]
    overview: list[dict[str, Any]]
    perinatal: list[dict[str, Any]]
    mortality: list[dict[str, Any]]
    maternal: list[dict[str, Any]]
    climate: list[dict[str, Any]]


class RelationshipFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_metric_id: str
    y_metric_id: str
    years: list[int] = Field(default_factory=list)
    granularity: Granularity = "month"
    geo_level: GeoLevel = "state"
    geo_codes: list[str] = Field(default_factory=list)
    x_value_mode: ValueMode | None = None
    y_value_mode: ValueMode | None = None
    context_variable: str | None = None
    context_bins: list[str] = Field(default_factory=list)

    @field_validator("years")
    @classmethod
    def _sort_relationship_years(cls, value: list[int]) -> list[int]:
        return sorted({int(item) for item in value})

    @field_validator("geo_codes", "context_bins")
    @classmethod
    def _normalize_relationship_lists(cls, value: list[str]) -> list[str]:
        return [str(item) for item in value if str(item)]


class RelationshipPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_value: float
    y_value: float
    geo_code: str
    label: str
    context_stratum: str | None = None
    time_bucket: str
    year: int
    quality_flag_any: bool
    tooltip_meta: dict[str, Any]


class TechnicalOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_count: int
    stratifier_count: int
    source_system_count: int
    years_covered: list[int]


class MedallionLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_id: str
    label: str
    description: str
    artifact_type: str


class SourceSystem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_system_id: str
    label: str
    short_description: str
    domain: str
    raw_manifest_ids: list[str]
    bronze_manifest_ids: list[str]
    silver_manifest_ids: list[str]
    notes: list[str]


class DisplayRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    title: str
    description: str


class TechnicalComponentVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    label: str
    role: str
    required_for: list[str] = Field(default_factory=list)


class TechnicalNumerator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    fields: list[str]
    logic: str


class TechnicalDenominator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    fields: list[str]
    logic: str
    minimum_threshold: int | None = None
    rate_formula: str | None = None


class IndicatorTechnicalSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet_id: str
    sheet_type: Literal["indicator"] = "indicator"
    label: str
    short_label: str
    metric_id: str
    metric_family: str
    page_id: str
    page_usage: list[str]
    relationship_compatible: bool
    used_as_kpi: bool
    definition_type: str
    display_unit: str
    rate_unit: str | None = None
    public_definition: str
    construction_logic: str
    aggregation_method: str
    allowed_geo_levels: list[str]
    allowed_granularities: list[str]
    available_value_modes: list[str]
    numerator: TechnicalNumerator
    denominator: TechnicalDenominator | None = None
    component_variables: list[TechnicalComponentVariable]
    source_system_ids: list[str]
    gold_dataset_id: str
    silver_entity_ids: list[str]
    bronze_family_ids: list[str]
    raw_source_ids: list[str]
    dashboard_artifact_id: str
    source_table: str
    numerator_field: str
    denominator_field: str | None = None
    quality_flags: list[str]
    display_rules: list[str]
    caveats: list[str]


class StratifierTechnicalSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet_id: str
    sheet_type: Literal["stratifier"] = "stratifier"
    label: str
    context_variable: str
    source_field: str
    stratifier_type: str
    applicable_pages: list[str]
    public_definition: str
    source_system_ids: list[str]
    observed_source_years: list[str]
    expanded_years: list[int]
    expansion_method: str
    join_key: str
    binning: str
    order: list[str]
    thresholds: list[float]
    threshold_labels: list[str]
    component_variables: list[TechnicalComponentVariable]
    binning_logic: str
    gold_dataset_id: str
    silver_entity_ids: list[str]
    bronze_family_ids: list[str]
    raw_source_ids: list[str]
    dashboard_artifact_id: str
    caveats: list[str]


class MethodsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: str
    technical_overview: TechnicalOverview
    medallion_strategy: dict[str, list[MedallionLayer]]
    source_systems: list[SourceSystem]
    display_rules: list[DisplayRule]
    indicator_sheets: list[IndicatorTechnicalSheet]
    stratifier_sheets: list[StratifierTechnicalSheet]
