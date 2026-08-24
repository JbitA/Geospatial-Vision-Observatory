from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .aoi import AOI
from .cells import (
    DEFAULT_CELL_PIXELS,
    DEFAULT_HALO_PIXELS,
    DEFAULT_RESOLUTION_M,
    ProcessingGrid,
    plan_processing_grid,
)
from .geodata import hansen_tile_ids_for_bbox, worldcover_tile_ids_for_bbox


@dataclass(frozen=True)
class PlannedAOI:
    aoi: AOI
    worldcover_tiles: tuple[str, ...]
    hansen_tiles: tuple[str, ...]
    processing_grid: ProcessingGrid
    legacy_executable: bool
    blockers: tuple[str, ...]


def plan_aoi(
    aoi: AOI,
    *,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    cell_pixels: int = DEFAULT_CELL_PIXELS,
    halo_pixels: int = DEFAULT_HALO_PIXELS,
) -> PlannedAOI:
    worldcover = tuple(sorted(worldcover_tile_ids_for_bbox(aoi.bbox)))
    hansen = tuple(sorted(hansen_tile_ids_for_bbox(aoi.bbox)))
    blockers: list[str] = []
    if len(worldcover) != 1:
        blockers.append("legacy engine requires one WorldCover tile")
    if len(hansen) != 1:
        blockers.append("legacy engine requires one Hansen tile")
    grid = plan_processing_grid(
        aoi,
        resolution_m=resolution_m,
        cell_pixels=cell_pixels,
        halo_pixels=halo_pixels,
    )
    return PlannedAOI(aoi, worldcover, hansen, grid, not blockers, tuple(blockers))


def build_plan(
    aois: list[AOI],
    *,
    recipe: str = "landcover-v1",
    resolution_m: float = DEFAULT_RESOLUTION_M,
    cell_pixels: int = DEFAULT_CELL_PIXELS,
    halo_pixels: int = DEFAULT_HALO_PIXELS,
) -> dict[str, Any]:
    planned = [
        plan_aoi(
            aoi,
            resolution_m=resolution_m,
            cell_pixels=cell_pixels,
            halo_pixels=halo_pixels,
        )
        for aoi in sorted(aois, key=lambda item: item.aoi_id)
    ]
    entries = [
        {
            "aoi_id": item.aoi.aoi_id,
            "geometry_id": item.aoi.geometry_id,
            "role": item.aoi.role.value,
            "bbox": list(item.aoi.bbox),
            "reference_tiles": {
                "worldcover": list(item.worldcover_tiles),
                "hansen": list(item.hansen_tiles),
            },
            "processing_grid": item.processing_grid.as_dict(),
            "legacy_executable": item.legacy_executable,
            "blockers": list(item.blockers),
        }
        for item in planned
    ]
    identity_payload = {
        "schema_version": "2.0",
        "recipe": recipe,
        "grid_contract": {
            "resolution_m": float(resolution_m),
            "cell_pixels": cell_pixels,
            "halo_pixels": halo_pixels,
        },
        "aois": entries,
    }
    encoded = json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    plan_id = hashlib.sha256(encoded).hexdigest()
    return {
        "schema_version": "2.0",
        "recipe": recipe,
        "plan_id": plan_id,
        "grid_contract": identity_payload["grid_contract"],
        "aois": entries,
    }
