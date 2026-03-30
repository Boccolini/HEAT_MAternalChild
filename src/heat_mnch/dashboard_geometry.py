from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import shapefile

from .dashboard_build import UF_TO_STATE_NAME, build_analytical_municipality_spine


def _point_line_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        return ((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5
    numerator = abs((end[1] - start[1]) * point[0] - (end[0] - start[0]) * point[1] + end[0] * start[1] - end[1] * start[0])
    denominator = ((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2) ** 0.5
    return numerator / denominator


def _rdp(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_distance = -1.0
    split_index = -1
    for index in range(1, len(points) - 1):
        distance = _point_line_distance(points[index], start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = index
    if max_distance <= tolerance:
        return [start, end]
    left = _rdp(points[: split_index + 1], tolerance)
    right = _rdp(points[split_index:], tolerance)
    return left[:-1] + right


def _simplify_ring(ring: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(ring) < 4:
        return ring
    closed = ring[0] == ring[-1]
    open_ring = ring[:-1] if closed else ring
    simplified = _rdp(open_ring, tolerance)
    if closed:
        simplified.append(simplified[0])
    return simplified if len(simplified) >= 4 else ring


def _shape_to_polygons(shape_geo: dict[str, Any], tolerance: float) -> list[list[list[float]]]:
    geometry_type = shape_geo["type"]
    coordinates = shape_geo["coordinates"]
    polygons: list[list[list[float]]] = []
    if geometry_type == "Polygon":
        polygon = []
        for ring in coordinates:
            simplified = _simplify_ring([tuple(point) for point in ring], tolerance)
            polygon.append([[float(x), float(y)] for x, y in simplified])
        polygons.append(polygon)
        return polygons
    if geometry_type == "MultiPolygon":
        for poly in coordinates:
            polygon = []
            for ring in poly:
                simplified = _simplify_ring([tuple(point) for point in ring], tolerance)
                polygon.append([[float(x), float(y)] for x, y in simplified])
            polygons.append(polygon)
        return polygons
    raise ValueError(f"Unsupported geometry type: {geometry_type}")


def _geometry_collection(features: list[dict[str, Any]], object_name: str) -> dict[str, Any]:
    arcs: list[list[list[float]]] = []
    geometries: list[dict[str, Any]] = []
    for feature in features:
        polygon_indexes: list[list[list[int]]] = []
        for polygon in feature["polygons"]:
            ring_indexes: list[list[int]] = []
            for ring in polygon:
                arc_index = len(arcs)
                arcs.append(ring)
                ring_indexes.append([arc_index])
            polygon_indexes.append(ring_indexes)
        geometry_type = "MultiPolygon" if len(polygon_indexes) > 1 else "Polygon"
        geometry_arcs: Any = polygon_indexes if geometry_type == "MultiPolygon" else polygon_indexes[0]
        geometries.append({"type": geometry_type, "properties": feature["properties"], "arcs": geometry_arcs})
    return {"type": "Topology", "objects": {object_name: {"type": "GeometryCollection", "geometries": geometries}}, "arcs": arcs}


def build_dashboard_geometry(root: Path, source_zip: Path, output_dir: Path, tolerance: float = 0.01) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analytical_spine = build_analytical_municipality_spine(root)
    with tempfile.TemporaryDirectory(prefix="observa_infancia_shape_") as tmpdir:
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(tmpdir)
        base = Path(tmpdir) / "BR_Municipios_2022"
        reader = shapefile.Reader(str(base.with_suffix(".shp")))
        municipal_features: list[dict[str, Any]] = []
        state_features: dict[str, dict[str, Any]] = {}

        for shape_record in reader.iterShapeRecords():
            code = str(shape_record.record["CD_MUN"]).zfill(7)
            if code not in analytical_spine:
                continue
            municipality_name = str(shape_record.record["NM_MUN"])
            uf = str(shape_record.record["SIGLA_UF"])
            polygons = _shape_to_polygons(shape_record.shape.__geo_interface__, tolerance)
            municipal_features.append(
                {
                    "properties": {
                        "geo_code": code,
                        "municipality_code": code,
                        "municipality_name": municipality_name,
                        "uf": uf,
                        "state_name": UF_TO_STATE_NAME[uf],
                    },
                    "polygons": polygons,
                }
            )
            state_feature = state_features.setdefault(
                uf,
                {
                    "properties": {
                        "geo_code": uf,
                        "uf": uf,
                        "state_name": UF_TO_STATE_NAME[uf],
                    },
                    "polygons": [],
                },
            )
            state_feature["polygons"].extend(polygons)

    if len(municipal_features) != len(analytical_spine):
        raise ValueError(
            "Simplified municipal geometry does not match the analytical municipality spine after filtering."
        )

    municipalities_topology = _geometry_collection(municipal_features, "municipalities")
    states_topology = _geometry_collection(list(state_features.values()), "states")

    municipalities_path = output_dir / "municipalities_2022_simplified.topojson"
    states_path = output_dir / "states_2022_simplified.topojson"
    municipalities_path.write_text(json.dumps(municipalities_topology), encoding="utf-8")
    states_path.write_text(json.dumps(states_topology), encoding="utf-8")
    return {
        "status": "PASS",
        "municipalities_path": str(municipalities_path),
        "states_path": str(states_path),
        "feature_counts": {
            "municipalities": len(municipal_features),
            "states": len(state_features),
        },
    }
