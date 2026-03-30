from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageDefinition:
    page_id: str
    route: str
    title: str
    description: str
    default_metric_id: str | None
    metric_ids: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageDefinition":
        return cls(
            page_id=str(payload["page_id"]),
            route=str(payload["route"]),
            title=str(payload["title"]),
            description=str(payload["description"]),
            default_metric_id=str(payload["default_metric_id"]) if payload.get("default_metric_id") else None,
            metric_ids=tuple(str(item) for item in payload.get("metric_ids", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    label: str
    short_label: str
    metric_family: str
    page_id: str
    source_table: str
    numerator_field: str
    denominator_field: str | None
    definition_type: str
    default_value_mode: str
    available_value_modes: tuple[str, ...]
    allowed_granularities: tuple[str, ...]
    allowed_geo_levels: tuple[str, ...]
    display_unit: str
    rate_display_unit: str | None
    minimum_denominator_threshold: int
    quality_flags: tuple[str, ...]
    tooltip_note: str
    aggregation_method: str
    aggregation_weight_field: str | None
    denominator_aggregation: str | None
    rate_multiplier: float | None
    sort_order: int
    technical_sheet: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricDefinition":
        return cls(
            metric_id=str(payload["metric_id"]),
            label=str(payload["label"]),
            short_label=str(payload.get("short_label", payload["label"])),
            metric_family=str(payload["metric_family"]),
            page_id=str(payload["page_id"]),
            source_table=str(payload["source_table"]),
            numerator_field=str(payload["numerator_field"]),
            denominator_field=str(payload["denominator_field"]) if payload.get("denominator_field") else None,
            definition_type=str(payload["definition_type"]),
            default_value_mode=str(payload["default_value_mode"]),
            available_value_modes=tuple(str(item) for item in payload.get("available_value_modes", [])),
            allowed_granularities=tuple(str(item) for item in payload.get("allowed_granularities", [])),
            allowed_geo_levels=tuple(str(item) for item in payload.get("allowed_geo_levels", [])),
            display_unit=str(payload["display_unit"]),
            rate_display_unit=str(payload["rate_display_unit"]) if payload.get("rate_display_unit") else None,
            minimum_denominator_threshold=int(payload.get("minimum_denominator_threshold", 0)),
            quality_flags=tuple(str(item) for item in payload.get("quality_flags", [])),
            tooltip_note=str(payload.get("tooltip_note", "")),
            aggregation_method=str(payload.get("aggregation_method", "sum")),
            aggregation_weight_field=str(payload["aggregation_weight_field"]) if payload.get("aggregation_weight_field") else None,
            denominator_aggregation=str(payload["denominator_aggregation"]) if payload.get("denominator_aggregation") else None,
            rate_multiplier=float(payload["rate_multiplier"]) if payload.get("rate_multiplier") is not None else None,
            sort_order=int(payload.get("sort_order", 0)),
            technical_sheet=payload.get("technical_sheet"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextVariableDefinition:
    context_variable: str
    label: str
    source_field: str
    binning: str
    missing_label: str
    order: tuple[str, ...]
    thresholds: tuple[float, ...] = field(default_factory=tuple)
    technical_sheet: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextVariableDefinition":
        return cls(
            context_variable=str(payload["context_variable"]),
            label=str(payload["label"]),
            source_field=str(payload["source_field"]),
            binning=str(payload["binning"]),
            missing_label=str(payload.get("missing_label", "missing")),
            order=tuple(str(item) for item in payload.get("order", [])),
            thresholds=tuple(float(item) for item in payload.get("thresholds", [])),
            technical_sheet=payload.get("technical_sheet"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSystemDefinition:
    source_system_id: str
    label: str
    short_description: str
    domain: str
    raw_manifest_ids: tuple[str, ...] = field(default_factory=tuple)
    bronze_manifest_ids: tuple[str, ...] = field(default_factory=tuple)
    silver_manifest_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceSystemDefinition":
        return cls(
            source_system_id=str(payload["source_system_id"]),
            label=str(payload["label"]),
            short_description=str(payload["short_description"]),
            domain=str(payload["domain"]),
            raw_manifest_ids=tuple(str(item) for item in payload.get("raw_manifest_ids", [])),
            bronze_manifest_ids=tuple(str(item) for item in payload.get("bronze_manifest_ids", [])),
            silver_manifest_ids=tuple(str(item) for item in payload.get("silver_manifest_ids", [])),
            notes=tuple(str(item) for item in payload.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MedallionLayerDefinition:
    layer_id: str
    label: str
    description: str
    artifact_type: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MedallionLayerDefinition":
        return cls(
            layer_id=str(payload["layer_id"]),
            label=str(payload["label"]),
            description=str(payload["description"]),
            artifact_type=str(payload["artifact_type"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoldDatasetLineageDefinition:
    gold_dataset_id: str
    source_system_ids: tuple[str, ...]
    bronze_family_ids: tuple[str, ...]
    silver_entity_ids: tuple[str, ...]
    gold_output: str
    dashboard_artifact_id: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoldDatasetLineageDefinition":
        return cls(
            gold_dataset_id=str(payload["gold_dataset_id"]),
            source_system_ids=tuple(str(item) for item in payload.get("source_system_ids", [])),
            bronze_family_ids=tuple(str(item) for item in payload.get("bronze_family_ids", [])),
            silver_entity_ids=tuple(str(item) for item in payload.get("silver_entity_ids", [])),
            gold_output=str(payload["gold_output"]),
            dashboard_artifact_id=str(payload["dashboard_artifact_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_like(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dashboard_config_dir(root: Path) -> Path:
    return root / "config" / "dashboard"


def load_page_registry(root: Path) -> list[PageDefinition]:
    path = dashboard_config_dir(root) / "indicator_registry.yaml"
    payload = _load_json_like(path)
    pages = payload.get("pages", [])
    return [PageDefinition.from_dict(item) for item in pages]


def load_metric_registry(root: Path) -> list[MetricDefinition]:
    path = dashboard_config_dir(root) / "indicator_registry.yaml"
    payload = _load_json_like(path)
    metrics = payload.get("metrics", [])
    return [MetricDefinition.from_dict(item) for item in metrics]


def load_context_registry(root: Path) -> list[ContextVariableDefinition]:
    path = dashboard_config_dir(root) / "context_registry.yaml"
    payload = _load_json_like(path)
    variables = payload.get("variables", [])
    return [ContextVariableDefinition.from_dict(item) for item in variables]


def load_source_system_registry(root: Path) -> list[SourceSystemDefinition]:
    path = dashboard_config_dir(root) / "source_registry.yaml"
    payload = _load_json_like(path)
    systems = payload.get("source_systems", [])
    return [SourceSystemDefinition.from_dict(item) for item in systems]


def load_medallion_layers(root: Path) -> list[MedallionLayerDefinition]:
    path = dashboard_config_dir(root) / "source_registry.yaml"
    payload = _load_json_like(path)
    layers = payload.get("medallion_layers", [])
    return [MedallionLayerDefinition.from_dict(item) for item in layers]


def load_gold_dataset_lineage(root: Path) -> list[GoldDatasetLineageDefinition]:
    path = dashboard_config_dir(root) / "source_registry.yaml"
    payload = _load_json_like(path)
    lineage = payload.get("gold_dataset_lineage", [])
    return [GoldDatasetLineageDefinition.from_dict(item) for item in lineage]


def metrics_by_id(root: Path) -> dict[str, MetricDefinition]:
    return {metric.metric_id: metric for metric in load_metric_registry(root)}


def pages_by_id(root: Path) -> dict[str, PageDefinition]:
    return {page.page_id: page for page in load_page_registry(root)}
