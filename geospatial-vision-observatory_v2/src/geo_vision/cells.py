from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Transformer
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from .aoi import AOI
from .geodata import hansen_tile_ids_for_bbox, worldcover_tile_ids_for_bbox

DEFAULT_RESOLUTION_M = 10.0
DEFAULT_CELL_PIXELS = 1024
DEFAULT_HALO_PIXELS = 32
MAX_PLANNED_CELLS = 100_000


@dataclass(frozen=True)
class WorkingCRS:
    authority: str | None
    definition: str
    strategy: str

    @property
    def identity(self) -> str:
        payload = f"{self.strategy}\n{self.definition}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProcessingCell:
    cell_id: str
    row: int
    column: int
    core_bounds_m: tuple[float, float, float, float]
    halo_bounds_m: tuple[float, float, float, float]
    wgs84_bbox: tuple[float, float, float, float]
    exact_geometry: dict[str, Any]
    coverage_fraction: float
    worldcover_tiles: tuple[str, ...]
    hansen_tiles: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "column": self.column,
            "core_bounds_m": list(self.core_bounds_m),
            "halo_bounds_m": list(self.halo_bounds_m),
            "wgs84_bbox": list(self.wgs84_bbox),
            "exact_geometry": self.exact_geometry,
            "coverage_fraction": self.coverage_fraction,
            "reference_tiles": {
                "worldcover": list(self.worldcover_tiles),
                "hansen": list(self.hansen_tiles),
            },
        }


@dataclass(frozen=True)
class ProcessingGrid:
    working_crs: WorkingCRS
    resolution_m: float
    cell_pixels: int
    halo_pixels: int
    cells: tuple[ProcessingCell, ...]

    @property
    def cell_size_m(self) -> float:
        return self.resolution_m * self.cell_pixels

    def as_dict(self) -> dict[str, Any]:
        return {
            "working_crs": {
                "authority": self.working_crs.authority,
                "definition": self.working_crs.definition,
                "strategy": self.working_crs.strategy,
                "crs_id": self.working_crs.identity,
            },
            "resolution_m": self.resolution_m,
            "cell_pixels": self.cell_pixels,
            "halo_pixels": self.halo_pixels,
            "cell_size_m": self.cell_size_m,
            "cells": [cell.as_dict() for cell in self.cells],
        }


def _same_utm_zone(west: float, east: float) -> int | None:
    if west < -180.0 or east > 180.0 or west >= east:
        return None
    west_zone = min(60, max(1, int(math.floor((west + 180.0) / 6.0)) + 1))
    east_probe = math.nextafter(east, west)
    east_zone = min(60, max(1, int(math.floor((east_probe + 180.0) / 6.0)) + 1))
    return west_zone if west_zone == east_zone else None


def select_working_crs(aoi: AOI) -> WorkingCRS:
    """Select one deterministic metre-based CRS for all processing cells in an AOI.

    Small AOIs that fit wholly inside one ordinary UTM zone use the corresponding EPSG CRS.
    Wider/multi-zone AOIs use an AOI-centred Lambert azimuthal equal-area CRS. Polar AOIs use
    standard UPS-compatible polar stereographic CRSs. Antimeridian AOIs are rejected earlier by
    the AOI contract, so the fallback centre is unambiguous.
    """

    geom = shape(aoi.geometry)
    centroid = geom.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    west, south, east, north = aoi.bbox

    if south >= 84.0:
        return WorkingCRS("EPSG:3413", "EPSG:3413", "polar_north")
    if north <= -80.0:
        return WorkingCRS("EPSG:3031", "EPSG:3031", "polar_south")

    zone = _same_utm_zone(west, east)
    if zone is not None and south >= -80.0 and north <= 84.0:
        epsg = (32600 if lat >= 0.0 else 32700) + zone
        token = f"EPSG:{epsg}"
        return WorkingCRS(token, token, "utm_single_zone")

    # Eight decimal places is far below sub-millimetre angular precision at Earth scale while
    # removing platform-specific floating representation noise from the serialized CRS contract.
    lat0 = round(lat, 8)
    lon0 = round(lon, 8)
    definition = (
        f"+proj=laea +lat_0={lat0:.8f} +lon_0={lon0:.8f} "
        "+datum=WGS84 +units=m +no_defs +type=crs"
    )
    # Constructing the CRS here fails closed if the local PROJ database/parser rejects it.
    CRS.from_user_input(definition)
    return WorkingCRS(None, definition, "laea_aoi_centered")


def _project_geometry(geometry: dict[str, Any], source: str, destination: str) -> BaseGeometry:
    transformer = Transformer.from_crs(source, destination, always_xy=True)
    projected = shapely_transform(transformer.transform, shape(geometry))
    if projected.is_empty or not projected.is_valid:
        raise ValueError("AOI cannot be represented safely in the selected working CRS")
    return projected


def _wgs84_geometry(
    projected: BaseGeometry, source: str, *, transformer: Transformer | None = None
) -> BaseGeometry:
    transformer = transformer or Transformer.from_crs(source, "EPSG:4326", always_xy=True)
    result = shapely_transform(transformer.transform, projected)
    if result.is_empty or not result.is_valid:
        raise ValueError("processing-cell geometry cannot be transformed back to WGS84")
    return result


def _stable_cell_id(
    aoi: AOI,
    working_crs: WorkingCRS,
    *,
    row: int,
    column: int,
    resolution_m: float,
    cell_pixels: int,
    halo_pixels: int,
) -> str:
    payload = {
        "aoi_geometry_id": aoi.geometry_id,
        "working_crs_id": working_crs.identity,
        "row": row,
        "column": column,
        "resolution_m": resolution_m,
        "cell_pixels": cell_pixels,
        "halo_pixels": halo_pixels,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_grid_parameters(
    resolution_m: float, cell_pixels: int, halo_pixels: int, max_cells: int
) -> None:
    if not math.isfinite(resolution_m) or not 1.0 <= resolution_m <= 1000.0:
        raise ValueError("resolution_m must be a finite value between 1 and 1000 metres")
    if isinstance(cell_pixels, bool) or not isinstance(cell_pixels, int) or not 64 <= cell_pixels <= 8192:
        raise ValueError("cell_pixels must be an integer between 64 and 8192")
    if isinstance(halo_pixels, bool) or not isinstance(halo_pixels, int) or not 0 <= halo_pixels <= 2048:
        raise ValueError("halo_pixels must be an integer between 0 and 2048")
    if halo_pixels * 2 >= cell_pixels:
        raise ValueError("halo_pixels must be less than half the core cell width")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or not 1 <= max_cells <= 1_000_000:
        raise ValueError("max_cells must be an integer between 1 and 1,000,000")


def plan_processing_grid(
    aoi: AOI,
    *,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    cell_pixels: int = DEFAULT_CELL_PIXELS,
    halo_pixels: int = DEFAULT_HALO_PIXELS,
    max_cells: int = MAX_PLANNED_CELLS,
) -> ProcessingGrid:
    """Partition an AOI into deterministic, non-overlapping projected core cells.

    The grid is anchored to the selected CRS origin rather than to the AOI bounds. Therefore small
    boundary edits do not renumber every cell. Only cells with positive-area AOI intersection are
    emitted. Halo bounds are contextual read bounds; scientific core bounds never overlap.
    """

    _validate_grid_parameters(resolution_m, cell_pixels, halo_pixels, max_cells)
    crs = select_working_crs(aoi)
    projected = _project_geometry(aoi.geometry, "EPSG:4326", crs.definition)
    minx, miny, maxx, maxy = projected.bounds
    if not all(math.isfinite(value) for value in (minx, miny, maxx, maxy)):
        raise ValueError("projected AOI bounds must be finite")

    cell_size = float(resolution_m) * cell_pixels
    min_column = math.floor(minx / cell_size)
    max_column = math.floor(math.nextafter(maxx, minx) / cell_size)
    min_row = math.floor(miny / cell_size)
    max_row = math.floor(math.nextafter(maxy, miny) / cell_size)
    candidate_count = (max_column - min_column + 1) * (max_row - min_row + 1)
    if candidate_count > max_cells * 8:
        raise ValueError(
            "AOI projected envelope creates too many candidate cells; reduce scope or increase cell size"
        )

    halo_m = float(resolution_m) * halo_pixels
    inverse = Transformer.from_crs(crs.definition, "EPSG:4326", always_xy=True)
    cells: list[ProcessingCell] = []
    for row in range(min_row, max_row + 1):
        bottom = row * cell_size
        top = bottom + cell_size
        for column in range(min_column, max_column + 1):
            left = column * cell_size
            right = left + cell_size
            core = box(left, bottom, right, top)
            clipped = projected.intersection(core)
            if clipped.is_empty or clipped.area <= 0.0:
                continue
            if len(cells) >= max_cells:
                raise ValueError(f"AOI exceeds the configured {max_cells:,}-cell planning limit")
            clipped_wgs84 = _wgs84_geometry(
                clipped, crs.definition, transformer=inverse
            ).normalize()
            exact_geometry = mapping(clipped_wgs84)
            wgs84_bbox = tuple(float(value) for value in clipped_wgs84.bounds)
            worldcover = tuple(sorted(worldcover_tile_ids_for_bbox(wgs84_bbox)))
            hansen = tuple(sorted(hansen_tile_ids_for_bbox(wgs84_bbox)))
            fraction = round(float(clipped.area / (cell_size * cell_size)), 12)
            cells.append(
                ProcessingCell(
                    cell_id=_stable_cell_id(
                        aoi,
                        crs,
                        row=row,
                        column=column,
                        resolution_m=float(resolution_m),
                        cell_pixels=cell_pixels,
                        halo_pixels=halo_pixels,
                    ),
                    row=row,
                    column=column,
                    core_bounds_m=(left, bottom, right, top),
                    halo_bounds_m=(
                        left - halo_m,
                        bottom - halo_m,
                        right + halo_m,
                        top + halo_m,
                    ),
                    wgs84_bbox=wgs84_bbox,
                    exact_geometry=exact_geometry,
                    coverage_fraction=fraction,
                    worldcover_tiles=worldcover,
                    hansen_tiles=hansen,
                )
            )
    if not cells:
        raise ValueError("AOI produced no positive-area processing cells")
    cells.sort(key=lambda cell: (-cell.row, cell.column, cell.cell_id))
    return ProcessingGrid(
        working_crs=crs,
        resolution_m=float(resolution_m),
        cell_pixels=cell_pixels,
        halo_pixels=halo_pixels,
        cells=tuple(cells),
    )


def cell_core_geometry(aoi: AOI, grid: ProcessingGrid, cell: ProcessingCell) -> dict[str, Any]:
    """Return the exact normalized WGS84 AOI intersection stored by the planner."""

    if cell not in grid.cells:
        raise ValueError("processing cell is not part of the supplied grid")
    cell_geom = shape(cell.exact_geometry)
    aoi_geom = shape(aoi.geometry)
    if cell_geom.is_empty or cell_geom.area <= 0.0 or not aoi_geom.covers(cell_geom.representative_point()):
        raise ValueError("processing-cell geometry is inconsistent with its AOI")
    return cell.exact_geometry
