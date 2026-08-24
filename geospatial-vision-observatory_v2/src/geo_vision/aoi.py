from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

_AOI_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class ScientificRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    EXTERNAL_UNOBSERVED = "external_unobserved"
    EXTERNAL_OBSERVED = "external_observed"
    INFERENCE_ONLY = "inference_only"


SCIENTIFIC_EVIDENCE_ROLES = frozenset(
    {
        ScientificRole.TRAIN,
        ScientificRole.VALIDATION,
        ScientificRole.EXTERNAL_UNOBSERVED,
        ScientificRole.EXTERNAL_OBSERVED,
    }
)


@dataclass(frozen=True)
class AOI:
    aoi_id: str
    name: str
    geometry: dict[str, Any]
    geometry_id: str
    bbox: tuple[float, float, float, float]
    role: ScientificRole
    purpose: str = ""

    def as_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "id": self.aoi_id,
            "geometry": self.geometry,
            "properties": {
                "name": self.name,
                "role": self.role.value,
                "purpose": self.purpose,
                "geometry_id": self.geometry_id,
            },
        }


_BASELINE_ROLES: dict[str, ScientificRole] = {
    "helsinki_metro": ScientificRole.TRAIN,
    "north_karelia_forest": ScientificRole.TRAIN,
    "turku_coast": ScientificRole.TRAIN,
    "oulu_mixed": ScientificRole.TRAIN,
    "tampere_growth": ScientificRole.VALIDATION,
    "jyvaskyla_validation": ScientificRole.VALIDATION,
    # These AOIs have already been evaluated in the baseline lineage.
    "stockholm_external": ScientificRole.EXTERNAL_OBSERVED,
    "tallinn_external": ScientificRole.EXTERNAL_OBSERVED,
}


def _validate_identifier(value: object) -> str:
    if not isinstance(value, str) or not _AOI_ID.fullmatch(value):
        raise ValueError(
            "AOI id must be 1-128 lowercase ASCII characters using letters, digits, '.', '_' or '-'"
        )
    return value


def _validated_geometry(value: Mapping[str, Any]) -> BaseGeometry:
    try:
        geom = shape(value)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("AOI geometry is not valid GeoJSON") from error
    if not isinstance(geom, Polygon | MultiPolygon):
        raise ValueError("AOI geometry must be a Polygon or MultiPolygon")
    if geom.is_empty:
        raise ValueError("AOI geometry must not be empty")
    if not geom.is_valid:
        raise ValueError(f"AOI geometry is invalid: {explain_validity(geom)}")
    minx, miny, maxx, maxy = geom.bounds
    if not all(math.isfinite(value) for value in (minx, miny, maxx, maxy)):
        raise ValueError("AOI geometry bounds must be finite")
    if minx < -180.0 or maxx > 180.0 or miny < -90.0 or maxy > 90.0:
        raise ValueError("AOI coordinates must be within WGS84 longitude/latitude bounds")
    # Iteration 1 deliberately fails closed for dateline-spanning shapes. Proper antimeridian
    # splitting belongs in the projected processing-cell planner rather than an implicit fix-up.
    if maxx - minx > 180.0:
        raise ValueError("AOI geometries spanning the antimeridian are not supported yet")
    if geom.area <= 0.0:
        raise ValueError("AOI geometry must have positive area")
    return geom


def canonical_geometry(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return normalized GeoJSON and a stable SHA-256 geometry identity.

    GEOS normalization makes equivalent polygon ring orientation/start ordering and
    MultiPolygon component order deterministic before hashing.
    """

    geom = _validated_geometry(value).normalize()
    canonical = mapping(geom)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    return canonical, digest


def aoi_from_feature(feature: Mapping[str, Any], *, default_role: ScientificRole | None = None) -> AOI:
    if feature.get("type") != "Feature":
        raise ValueError("AOI document must be a GeoJSON Feature")
    properties = feature.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, Mapping):
        raise ValueError("AOI Feature properties must be an object")
    raw_id = feature.get("id", properties.get("id"))
    aoi_id = _validate_identifier(raw_id)
    raw_name = properties.get("name", aoi_id)
    if not isinstance(raw_name, str) or not raw_name.strip() or len(raw_name) > 256:
        raise ValueError("AOI name must be a non-empty string up to 256 characters")
    raw_purpose = properties.get("purpose", "")
    if not isinstance(raw_purpose, str) or len(raw_purpose) > 1000:
        raise ValueError("AOI purpose must be a string up to 1000 characters")
    role_value = properties.get("role")
    if role_value is None:
        if default_role is None:
            raise ValueError("AOI properties.role is required")
        role = default_role
    else:
        try:
            role = ScientificRole(str(role_value))
        except ValueError as error:
            raise ValueError(f"unsupported AOI scientific role: {role_value!r}") from error
    raw_geometry = feature.get("geometry")
    if not isinstance(raw_geometry, Mapping):
        raise ValueError("AOI Feature geometry must be a GeoJSON object")
    geometry, geometry_id = canonical_geometry(raw_geometry)
    geom = shape(geometry)
    return AOI(
        aoi_id=aoi_id,
        name=raw_name.strip(),
        geometry=geometry,
        geometry_id=geometry_id,
        bbox=tuple(float(value) for value in geom.bounds),
        role=role,
        purpose=raw_purpose,
    )


def aoi_from_bbox(
    aoi_id: str,
    bbox: tuple[float, float, float, float],
    *,
    name: str | None = None,
    role: ScientificRole = ScientificRole.INFERENCE_ONLY,
    purpose: str = "",
) -> AOI:
    _validate_identifier(aoi_id)
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        raise ValueError("bbox must contain four finite numbers")
    west, south, east, north = (float(value) for value in bbox)
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError("bbox must be an ordered WGS84 extent")
    feature = {
        "type": "Feature",
        "id": aoi_id,
        "geometry": mapping(box(west, south, east, north)),
        "properties": {
            "name": name or aoi_id,
            "role": role.value,
            "purpose": purpose,
        },
    }
    return aoi_from_feature(feature)


def baseline_aoi(aoi_id: str, legacy: Any) -> AOI:
    role = _BASELINE_ROLES.get(aoi_id)
    if role is None:
        raise ValueError(f"baseline AOI has no declared scientific role: {aoi_id}")
    return aoi_from_bbox(
        aoi_id,
        tuple(float(value) for value in legacy.bbox),
        name=str(legacy.name),
        role=role,
        purpose=str(legacy.purpose),
    )


def validate_scientific_isolation(aois: list[AOI]) -> None:
    """Reject positive-area overlap between AOIs used as independent scientific evidence.

    Inference-only AOIs are allowed to overlap because they do not contribute independent
    train/validation/test evidence. Scientific areas that intentionally consist of several
    disconnected pieces should be represented by one MultiPolygon AOI.
    """

    seen_ids: set[str] = set()
    seen_geometry_ids: set[str] = set()
    for aoi in aois:
        if aoi.aoi_id in seen_ids:
            raise ValueError(f"duplicate AOI id: {aoi.aoi_id}")
        seen_ids.add(aoi.aoi_id)
        if aoi.role in SCIENTIFIC_EVIDENCE_ROLES:
            if aoi.geometry_id in seen_geometry_ids:
                raise ValueError(f"duplicate scientific geometry: {aoi.aoi_id}")
            seen_geometry_ids.add(aoi.geometry_id)

    scientific = [aoi for aoi in aois if aoi.role in SCIENTIFIC_EVIDENCE_ROLES]
    for index, left in enumerate(scientific):
        left_geom = shape(left.geometry)
        for right in scientific[index + 1 :]:
            intersection = left_geom.intersection(shape(right.geometry))
            if not intersection.is_empty and intersection.area > 0.0:
                raise ValueError(
                    "scientific AOIs must be positive-area disjoint: "
                    f"{left.aoi_id} ({left.role.value}) overlaps "
                    f"{right.aoi_id} ({right.role.value})"
                )


def load_aoi_document(payload: Mapping[str, Any]) -> list[AOI]:
    doc_type = payload.get("type")
    if doc_type == "Feature":
        result = [aoi_from_feature(payload)]
    elif doc_type == "FeatureCollection":
        raw_features = payload.get("features")
        if not isinstance(raw_features, list) or not raw_features:
            raise ValueError("AOI FeatureCollection must contain at least one Feature")
        if len(raw_features) > 10000:
            raise ValueError("AOI FeatureCollection exceeds the 10,000 AOI local planning limit")
        result = []
        for feature in raw_features:
            if not isinstance(feature, Mapping):
                raise ValueError("AOI FeatureCollection entries must be GeoJSON Features")
            result.append(aoi_from_feature(feature))
    else:
        raise ValueError("AOI document must be a GeoJSON Feature or FeatureCollection")
    validate_scientific_isolation(result)
    return result
