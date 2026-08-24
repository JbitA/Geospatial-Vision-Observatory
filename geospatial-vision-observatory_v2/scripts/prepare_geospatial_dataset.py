#!/usr/bin/env python3
"""Build a compact Sentinel-2 + WorldCover + Hansen urban/forestry dataset.

The command reads Cloud-Optimized GeoTIFF windows rather than downloading global datasets.
It creates a six-band Sentinel-2 stack, spectral indices, WorldCover labels and Hansen forestry
layers on one common grid, plus a preview and a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
from PIL import Image
from shapely.geometry import box as shapely_box, mapping as shapely_mapping, shape as shapely_shape
from shapely.ops import unary_union

try:
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.warp import reproject, transform_bounds, transform_geom
    from rasterio.windows import from_bounds as window_from_bounds
except ImportError as error:  # pragma: no cover - runtime guard
    raise SystemExit("Install the geospatial extra: pip install -e '.[geo]'") from error

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from geo_vision import __version__  # noqa: E402
from geo_vision.aoi import AOI, aoi_from_feature, baseline_aoi  # noqa: E402
from geo_vision.cells import ProcessingCell, ProcessingGrid, cell_core_geometry, plan_processing_grid  # noqa: E402
from geo_vision.planning import plan_aoi  # noqa: E402
from geo_vision.raster_cells import (  # noqa: E402
    cell_aoi_mask,
    cell_profile,
    reproject_calibrated_sources,
    reproject_sources,
    reproject_sources_with_coverage,
)
from geo_vision.resume import (  # noqa: E402
    CELL_DATASET_LAYOUT,
    CELL_MANIFEST_SCHEMA_VERSION,
    COLLECTION_DATASET_LAYOUT,
    COLLECTION_SCHEMA_VERSION,
    RUN_STATE_SCHEMA_VERSION,
    atomic_write_json,
    canonical_sha256,
    cell_recipe_identity,
    load_json_object,
    parse_utc,
    prior_source_selection_id,
    request_identity,
    resolve_run_window,
    run_identity,
    source_selection_identity,
    utc_iso,
    validate_processing_cell_manifest,
)
from geo_vision.geodata import (  # noqa: E402
    CURATED_AOIS,
    WORLDCOVER_CLASSES,
    bbox_center,
    hansen_url,
    hansen_url_for_tile,
    spectral_indices,
    worldcover_map_url,
    worldcover_map_url_for_tile,
)

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ALLOWED_ASSET_HOSTS = {
    "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
}
MAX_STAC_BYTES = 8 * 1024 * 1024
STAC_ITEM_LIMIT = 200
MAX_SCL_QUALITY_CANDIDATES = 24
SENTINEL_BANDS = ("red", "green", "blue", "nir", "swir16", "swir22")
SCL_EXCLUDED_CLASSES = frozenset({0, 1, 3, 7, 8, 9, 10})


def _validate_https_url(value: object, allowed_hosts: set[str], *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} URL is missing")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError(f"{label} URL violates HTTPS/host policy")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} URL must not contain userinfo")
    if parsed.port not in (None, 443):
        raise ValueError(f"{label} URL must use the default HTTPS port")
    if parsed.fragment:
        raise ValueError(f"{label} URL must not contain a fragment")
    return value


def validated_asset_href(value: object) -> str:
    return _validate_https_url(value, ALLOWED_ASSET_HOSTS, label="Sentinel asset")


def bounded_json_get(url: str, params: dict[str, str]) -> dict[str, Any]:
    _validate_https_url(url, {"earth-search.aws.element84.com"}, label="STAC endpoint")
    headers = {
        "Accept": "application/geo+json,application/json",
        "User-Agent": "geospatial-vision-observatory/1.0 dataset-builder",
    }
    chunks: list[bytes] = []
    received = 0
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        with client.stream("GET", url, params=params, headers=headers) as response:
            if 300 <= response.status_code < 400:
                raise RuntimeError("STAC redirects are rejected")
            response.raise_for_status()
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type not in {"application/json", "application/geo+json"}:
                raise RuntimeError("STAC response has an unexpected media type")
            declared = response.headers.get("content-length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as error:
                    raise RuntimeError("STAC response has invalid Content-Length") from error
                if declared_bytes < 0:
                    raise RuntimeError("STAC response has invalid Content-Length")
                if declared_bytes > MAX_STAC_BYTES:
                    raise RuntimeError("STAC metadata exceeds the size ceiling")
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > MAX_STAC_BYTES:
                    raise RuntimeError("STAC metadata exceeds the size ceiling")
                chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks))
    except json.JSONDecodeError as error:
        raise RuntimeError("STAC response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("STAC response must be a JSON object")
    return payload


def scl_obscured_percent(
    href: str,
    bbox: tuple[float, float, float, float],
    *,
    geometry: dict[str, Any] | None = None,
    max_pixels: int = 512,
) -> float:
    """Measure SCL-obscured pixels over the requested AOI, not the whole granule."""

    if not 64 <= max_pixels <= 2048:
        raise ValueError("SCL quality max_pixels must be between 64 and 2048")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise RuntimeError("Sentinel SCL COG is missing a CRS")
            projected = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = (
                window_from_bounds(*projected, transform=src.transform)
                .round_offsets()
                .round_lengths()
            )
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            width = max(1, int(window.width))
            height = max(1, int(window.height))
            scale = max(width / max_pixels, height / max_pixels, 1.0)
            out_width = max(1, int(round(width / scale)))
            out_height = max(1, int(round(height / scale)))
            scl = src.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=Resampling.nearest,
            )
            inside = np.ones(scl.shape, dtype=bool)
            if geometry is not None:
                projected_geometry = transform_geom("EPSG:4326", src.crs, geometry)
                out_transform = src.window_transform(window) * Affine.scale(
                    float(window.width) / out_width, float(window.height) / out_height
                )
                inside = geometry_mask(
                    [projected_geometry],
                    out_shape=scl.shape,
                    transform=out_transform,
                    invert=True,
                )
    if scl.size == 0 or not inside.any():
        raise RuntimeError("Sentinel SCL AOI window is empty")
    obscured = np.isin(scl[inside], list(SCL_EXCLUDED_CLASSES))
    return round(float(obscured.mean() * 100.0), 6)


def stac_item(
    bbox: tuple[float, float, float, float],
    lookback_days: int,
    max_cloud: float,
    *,
    geometry: dict[str, Any] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cloud_threshold: bool = False,
    quality_max_pixels: int = 512,
) -> dict[str, Any]:
    if not 1 <= lookback_days <= 3650:
        raise ValueError("lookback_days must be between 1 and 3650")
    if not math.isfinite(max_cloud) or not 0.0 <= max_cloud <= 100.0:
        raise ValueError("max_cloud must be a finite percentage between 0 and 100")
    now = datetime.now(UTC)
    end = end or now
    start = start or (end - timedelta(days=lookback_days))
    for label, value in (("start", start), ("end", end)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"Sentinel search {label} must be timezone-aware")
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    if start >= end:
        raise ValueError("Sentinel search start must be before end")
    params = {
        "bbox": ",".join(str(value) for value in bbox),
        "datetime": (
            f"{start.isoformat().replace('+00:00', 'Z')}/"
            f"{end.isoformat().replace('+00:00', 'Z')}"
        ),
        # Earth Search documents sorting on item properties. Asking for a bounded, larger page
        # prevents a long seasonal window from accidentally considering only the newest 50 items.
        "limit": str(STAC_ITEM_LIMIT),
        "sortby": "+properties.eo:cloud_cover,+properties.datetime",
    }
    payload = bounded_json_get(EARTH_SEARCH, params)
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise RuntimeError("STAC response has an invalid features collection")
    candidates: list[tuple[datetime, float, dict[str, Any]]] = []
    for item in features:
        if not isinstance(item, dict):
            continue
        properties = item.get("properties", {})
        assets = item.get("assets", {})
        if not isinstance(properties, dict) or not isinstance(assets, dict):
            continue
        required = {name: assets.get(name, {}) for name in SENTINEL_BANDS}
        if any(not isinstance(asset, dict) for asset in required.values()):
            continue
        try:
            for asset in required.values():
                validated_asset_href(asset.get("href"))
        except ValueError:
            continue
        if not isinstance(item.get("id"), str) or not item["id"]:
            continue
        stamp = properties.get("datetime")
        if not isinstance(stamp, str):
            continue
        try:
            acquired = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if acquired.tzinfo is None or acquired.utcoffset() is None:
                continue
            acquired = acquired.astimezone(UTC)
            cloud = float(properties.get("eo:cloud_cover", 100.0))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(cloud) or not 0.0 <= cloud <= 100.0:
            continue
        if not start <= acquired <= end:
            continue
        candidates.append((acquired, cloud, item))
    if not candidates:
        raise RuntimeError("Earth Search returned no usable Sentinel-2 multispectral COGs")

    midpoint = start + (end - start) / 2
    candidates.sort(
        key=lambda row: (
            row[1],
            abs((row[0] - midpoint).total_seconds()),
            -row[0].timestamp(),
        )
    )
    if not require_cloud_threshold:
        acceptable = [row for row in candidates if row[1] <= max_cloud]
        return (acceptable or candidates)[0][2]

    # eo:cloud_cover describes the STAC item/granule. For a small AOI it is only a discovery
    # heuristic; publication quality is measured from the Sentinel Scene Classification Layer
    # over the actual AOI. This permits a clear AOI inside an otherwise cloudy granule without
    # weakening the 15% quality gate.
    quality_rows: list[tuple[float, float, float, dict[str, Any]]] = []
    inspected = 0
    for acquired, scene_cloud, item in candidates[:MAX_SCL_QUALITY_CANDIDATES]:
        assets = item.get("assets", {})
        if not isinstance(assets, dict):
            continue
        href = scl_asset_href(assets)
        if href is None:
            continue
        inspected += 1
        try:
            obscured = scl_obscured_percent(
                href,
                bbox,
                geometry=geometry,
                max_pixels=min(max(quality_max_pixels, 64), 2048),
            )
        except (OSError, RuntimeError, rasterio.errors.RasterioError):
            continue
        quality_rows.append(
            (
                obscured,
                scene_cloud,
                abs((acquired - midpoint).total_seconds()),
                item,
            )
        )
    if not quality_rows:
        raise RuntimeError(
            "no Sentinel-2 candidate exposed a readable SCL layer for AOI cloud-quality checks "
            f"(inspected {inspected} candidates)"
        )
    acceptable_quality = [row for row in quality_rows if row[0] <= max_cloud]
    if not acceptable_quality:
        best = min(quality_rows, key=lambda row: (row[0], row[1], row[2]))
        raise RuntimeError(
            "no Sentinel-2 scene meets the required AOI SCL obscured-pixel threshold "
            f"({max_cloud}%); best AOI obscured fraction was {best[0]}% "
            f"with scene-wide eo:cloud_cover={best[1]}%"
        )
    selected = min(acceptable_quality, key=lambda row: (row[0], row[1], row[2]))
    chosen = dict(selected[3])
    properties = dict(chosen.get("properties", {}))
    properties["gvo:aoi_scl_obscured_percent"] = selected[0]
    chosen["properties"] = properties
    return chosen




def _stac_geometry(item: dict[str, Any]) -> Any | None:
    raw = item.get("geometry")
    if isinstance(raw, dict):
        try:
            geom = shapely_shape(raw)
        except (TypeError, ValueError, KeyError):
            return None
        if not geom.is_empty and geom.is_valid:
            return geom
    raw_bbox = item.get("bbox")
    if (
        isinstance(raw_bbox, list)
        and len(raw_bbox) >= 4
        and all(isinstance(value, int | float) and not isinstance(value, bool) for value in raw_bbox[:4])
    ):
        west, south, east, north = (float(value) for value in raw_bbox[:4])
        if west < east and south < north:
            return shapely_box(west, south, east, north)
    return None


def stac_item_set(
    bbox: tuple[float, float, float, float],
    lookback_days: int,
    max_cloud: float,
    *,
    geometry: dict[str, Any] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cloud_threshold: bool = False,
    quality_max_pixels: int = 512,
    max_time_span_minutes: int = 30,
) -> list[dict[str, Any]]:
    """Select a deterministic same-overpass Sentinel item set that covers the target geometry.

    Iteration 2 uses this only for bounded processing cells. Candidate items are clustered by
    acquisition time, then a deterministic greedy set-cover chooses the smallest useful subset.
    A cluster is accepted only when its STAC footprints cover the full target geometry. This fails
    closed rather than silently accepting a partially covered cell at a Sentinel tile boundary.
    """

    if not 1 <= lookback_days <= 3650:
        raise ValueError("lookback_days must be between 1 and 3650")
    if not math.isfinite(max_cloud) or not 0.0 <= max_cloud <= 100.0:
        raise ValueError("max_cloud must be a finite percentage between 0 and 100")
    if not 1 <= max_time_span_minutes <= 120:
        raise ValueError("max_time_span_minutes must be between 1 and 120")
    now = datetime.now(UTC)
    end = end or now
    start = start or (end - timedelta(days=lookback_days))
    for label, value in (("start", start), ("end", end)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"Sentinel search {label} must be timezone-aware")
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    if start >= end:
        raise ValueError("Sentinel search start must be before end")

    target = shapely_shape(geometry) if geometry is not None else shapely_box(*bbox)
    if target.is_empty or not target.is_valid or target.area <= 0.0:
        raise ValueError("Sentinel cell target geometry must have positive area")
    midpoint = start + (end - start) / 2
    params = {
        "bbox": ",".join(str(value) for value in bbox),
        "datetime": (
            f"{start.isoformat().replace('+00:00', 'Z')}/"
            f"{end.isoformat().replace('+00:00', 'Z')}"
        ),
        "limit": str(STAC_ITEM_LIMIT),
        "sortby": "+properties.eo:cloud_cover,+properties.datetime",
    }
    payload = bounded_json_get(EARTH_SEARCH, params)
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        raise RuntimeError("STAC response has an invalid features collection")

    rows: list[tuple[datetime, float, str, dict[str, Any], Any]] = []
    for raw_item in raw_features:
        if not isinstance(raw_item, dict):
            continue
        properties = raw_item.get("properties", {})
        assets = raw_item.get("assets", {})
        if not isinstance(properties, dict) or not isinstance(assets, dict):
            continue
        item_id = raw_item.get("id")
        stamp = properties.get("datetime")
        if not isinstance(item_id, str) or not item_id or not isinstance(stamp, str):
            continue
        required = {name: assets.get(name, {}) for name in SENTINEL_BANDS}
        if any(not isinstance(asset, dict) for asset in required.values()):
            continue
        try:
            for asset in required.values():
                validated_asset_href(asset.get("href"))
            acquired = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            cloud = float(properties.get("eo:cloud_cover", 100.0))
        except (TypeError, ValueError, OverflowError):
            continue
        if acquired.tzinfo is None or acquired.utcoffset() is None:
            continue
        acquired = acquired.astimezone(UTC)
        if not start <= acquired <= end or not math.isfinite(cloud) or not 0.0 <= cloud <= 100.0:
            continue
        footprint = _stac_geometry(raw_item)
        if footprint is None:
            continue
        overlap = footprint.intersection(target)
        if overlap.is_empty or overlap.area <= 0.0:
            continue
        rows.append((acquired, cloud, item_id, raw_item, overlap))
    if not rows:
        raise RuntimeError("Earth Search returned no usable Sentinel-2 items intersecting the cell")

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    span = timedelta(minutes=max_time_span_minutes)
    clusters: list[list[tuple[datetime, float, str, dict[str, Any], Any]]] = []
    seen_windows: set[tuple[str, ...]] = set()
    for start_index, first in enumerate(rows):
        cluster = [
            row
            for row in rows[start_index:]
            if row[0] - first[0] <= span
        ]
        identity = tuple(row[2] for row in cluster)
        if identity and identity not in seen_windows:
            seen_windows.add(identity)
            clusters.append(cluster)

    solutions: list[tuple[tuple[float, float, float, int, str], list[dict[str, Any]]]] = []
    target_area = target.area
    tolerance = max(target_area * 1e-9, 1e-15)
    for cluster in clusters:
        usable: list[tuple[float, float, str, datetime, dict[str, Any], Any]] = []
        for acquired, scene_cloud, item_id, raw_item, overlap in cluster:
            quality = scene_cloud
            chosen_item = raw_item
            if require_cloud_threshold:
                assets = raw_item.get("assets", {})
                if not isinstance(assets, dict):
                    continue
                scl_href = scl_asset_href(assets)
                if scl_href is None:
                    continue
                overlap_geometry = shapely_mapping(overlap)
                try:
                    quality = scl_obscured_percent(
                        scl_href,
                        tuple(float(value) for value in overlap.bounds),
                        geometry=overlap_geometry,
                        max_pixels=min(max(quality_max_pixels, 64), 2048),
                    )
                except (OSError, RuntimeError, rasterio.errors.RasterioError):
                    continue
                if quality > max_cloud:
                    continue
                chosen_item = dict(raw_item)
                updated_properties = dict(chosen_item.get("properties", {}))
                updated_properties["gvo:cell_scl_obscured_percent"] = quality
                chosen_item["properties"] = updated_properties
            elif scene_cloud > max_cloud:
                continue
            usable.append((quality, scene_cloud, item_id, acquired, chosen_item, overlap))
        if not usable:
            continue
        usable.sort(key=lambda row: (row[0], row[1], abs((row[3] - midpoint).total_seconds()), row[2]))
        selected: list[dict[str, Any]] = []
        covered_parts: list[Any] = []
        covered = None
        remaining = target
        while remaining.area > tolerance:
            best = None
            best_gain = tolerance
            for row in usable:
                if row[4] in selected:
                    continue
                gain = row[5].intersection(remaining).area
                if gain > best_gain:
                    best_gain = gain
                    best = row
            if best is None:
                break
            selected.append(best[4])
            covered_parts.append(best[5])
            covered = unary_union(covered_parts)
            remaining = target.difference(covered)
        if remaining.area > tolerance or not selected:
            continue
        selected_ids = ",".join(str(item["id"]) for item in selected)
        qualities = [
            float(item.get("properties", {}).get("gvo:cell_scl_obscured_percent", item.get("properties", {}).get("eo:cloud_cover", 100.0)))
            for item in selected
        ]
        acquired_times = [
            datetime.fromisoformat(str(item["properties"]["datetime"]).replace("Z", "+00:00")).astimezone(UTC)
            for item in selected
        ]
        score = (
            max(qualities),
            sum(qualities) / len(qualities),
            abs(((min(acquired_times) + (max(acquired_times) - min(acquired_times)) / 2) - midpoint).total_seconds()),
            len(selected),
            selected_ids,
        )
        solutions.append((score, selected))
    if not solutions:
        raise RuntimeError(
            "no same-overpass Sentinel-2 item set fully covers the processing cell within the cloud policy"
        )
    solutions.sort(key=lambda row: row[0])
    return solutions[0][1]


def asset_calibration(asset: dict[str, Any]) -> dict[str, float | int | None]:
    """Extract STAC raster scale, offset and nodata without assuming processing baseline."""

    raster_bands = asset.get("raster:bands")
    metadata = raster_bands[0] if isinstance(raster_bands, list) and raster_bands else {}
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        scale = float(metadata.get("scale", 1.0))
        offset = float(metadata.get("offset", 0.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Sentinel raster calibration is not numeric") from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Sentinel raster scale must be finite and positive")
    if not math.isfinite(offset):
        raise ValueError("Sentinel raster offset must be finite")
    nodata_value = metadata.get("nodata", 0)
    nodata: float | None = None
    if isinstance(nodata_value, int | float) and not isinstance(nodata_value, bool):
        nodata = float(nodata_value)
        if not math.isfinite(nodata):
            raise ValueError("Sentinel raster nodata must be finite")
    return {"scale": scale, "offset": offset, "nodata": nodata}


def calibrated_reflectance(array: np.ndarray, asset: dict[str, Any]) -> np.ndarray:
    calibration = asset_calibration(asset)
    result = array.astype(np.float32) * float(calibration["scale"]) + float(
        calibration["offset"]
    )
    nodata = calibration["nodata"]
    if isinstance(nodata, int | float):
        result[array == nodata] = np.nan
    return result


def scl_asset_href(assets: dict[str, Any]) -> str | None:
    for key in ("scl", "SCL"):
        asset = assets.get(key)
        if isinstance(asset, dict) and isinstance(asset.get("href"), str):
            try:
                return validated_asset_href(asset["href"])
            except ValueError:
                continue
    return None


def read_reference_window(
    href: str,
    bbox: tuple[float, float, float, float],
    max_pixels: int = 1536,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read band 1 from a COG and return it with an AOI-scoped target profile."""

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise RuntimeError("source COG is missing a CRS")
            projected = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = (
                window_from_bounds(*projected, transform=src.transform)
                .round_offsets()
                .round_lengths()
            )
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            width = max(1, int(window.width))
            height = max(1, int(window.height))
            scale = max(width / max_pixels, height / max_pixels, 1.0)
            out_width = max(1, int(round(width / scale)))
            out_height = max(1, int(round(height / scale)))
            array = src.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=Resampling.bilinear,
            )
            transform = src.window_transform(window) * Affine.scale(
                window.width / out_width, window.height / out_height
            )
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                width=out_width,
                height=out_height,
                count=1,
                transform=transform,
                compress="deflate",
                tiled=True,
            )
            return array, profile


def write_tif(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    descriptions: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=array.shape[0])
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array)
        if descriptions:
            if len(descriptions) != array.shape[0]:
                raise ValueError("band descriptions do not match output count")
            for index, description in enumerate(descriptions, start=1):
                dst.set_band_description(index, description)


def reproject_single_band(
    href: str,
    destination_profile: dict[str, Any],
    *,
    dtype: np.dtype[Any],
    resampling: Resampling,
) -> np.ndarray:
    destination = np.zeros(
        (destination_profile["height"], destination_profile["width"]), dtype=dtype
    )
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise RuntimeError("source COG is missing a CRS")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=destination_profile["transform"],
                dst_crs=destination_profile["crs"],
                dst_nodata=0,
                resampling=resampling,
            )
    return destination


def robust_rgb_preview(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.stack((red, green, blue)).astype(np.float32)
    preview = np.zeros_like(rgb, dtype=np.uint8)
    for index in range(3):
        band = rgb[index]
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size == 0:
            continue
        low, high = np.quantile(valid, [0.02, 0.98])
        stretched = np.clip((band - low) / max(float(high - low), 1e-6), 0, 1)
        stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
        preview[index] = (stretched * 255).astype(np.uint8)
    return np.transpose(preview, (1, 2, 0))


def class_stats(labels: np.ndarray) -> dict[str, float]:
    valid = labels != 0
    denominator = int(valid.sum())
    if denominator == 0:
        return {}
    return {
        name: round(float((labels == code).sum() / denominator), 6)
        for code, name in WORLDCOVER_CLASSES.items()
        if np.any(labels == code)
    }


def index_stats(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {}
    return {
        "mean": round(float(valid.mean()), 6),
        "median": round(float(np.median(valid)), 6),
        "p05": round(float(np.quantile(valid, 0.05)), 6),
        "p95": round(float(np.quantile(valid, 0.95)), 6),
    }


def aoi_mask_on_profile(aoi: AOI, profile: dict[str, Any]) -> np.ndarray:
    projected = transform_geom("EPSG:4326", profile["crs"], aoi.geometry)
    mask = geometry_mask(
        [projected],
        out_shape=(int(profile["height"]), int(profile["width"])),
        transform=profile["transform"],
        invert=True,
    )
    if not mask.any():
        raise RuntimeError("AOI geometry has no pixels on the selected Sentinel grid")
    return mask


def build_dataset(
    aoi_key: str,
    output_root: Path,
    lookback_days: int,
    max_cloud: float,
    *,
    aoi_spec: AOI | None = None,
    max_pixels: int = 1024,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cloud_threshold: bool = False,
) -> Path:
    if not 1 <= lookback_days <= 3650:
        raise ValueError("lookback_days must be between 1 and 3650")
    if not math.isfinite(max_cloud) or not 0.0 <= max_cloud <= 100.0:
        raise ValueError("max_cloud must be a finite percentage between 0 and 100")
    if not 256 <= max_pixels <= 2048:
        raise ValueError("max_pixels must be between 256 and 2048")
    if (start is None) != (end is None):
        raise ValueError("start and end must be provided together")
    aoi = aoi_spec or baseline_aoi(aoi_key, CURATED_AOIS[aoi_key])
    if aoi.aoi_id != aoi_key:
        raise ValueError("AOI key and AOI specification id must match")
    execution_plan = plan_aoi(aoi)
    if not execution_plan.legacy_executable:
        raise ValueError(
            f"AOI {aoi.aoi_id!r} requires the Iteration 2 multi-tile processing-cell engine: "
            + "; ".join(execution_plan.blockers)
        )
    item = stac_item(
        aoi.bbox,
        lookback_days,
        max_cloud,
        geometry=aoi.geometry,
        start=start,
        end=end,
        require_cloud_threshold=require_cloud_threshold,
        quality_max_pixels=min(max_pixels, 512),
    )
    assets = item["assets"]
    if not isinstance(assets, dict):
        raise RuntimeError("selected STAC item has no valid asset map")
    band_hrefs = {name: validated_asset_href(assets[name]["href"]) for name in SENTINEL_BANDS}

    red, profile = read_reference_window(band_hrefs["red"], aoi.bbox, max_pixels=max_pixels)
    profile.update(dtype=str(red.dtype), nodata=0)
    raw_bands: dict[str, np.ndarray] = {"red": red}
    for name in SENTINEL_BANDS:
        if name == "red":
            continue
        raw_bands[name] = reproject_single_band(
            band_hrefs[name],
            profile,
            dtype=red.dtype,
            resampling=Resampling.bilinear,
        )
    reflectance = {
        name: calibrated_reflectance(raw_bands[name], assets[name])
        for name in SENTINEL_BANDS
    }
    inside_aoi = aoi_mask_on_profile(aoi, profile)
    reflectance = {
        name: np.where(inside_aoi, values, np.nan).astype(np.float32)
        for name, values in reflectance.items()
    }

    scl_href = scl_asset_href(assets)
    scl: np.ndarray | None = None
    scl_valid_fraction: float | None = None
    scl_obscured: float | None = None
    if scl_href is not None:
        scl = reproject_single_band(
            scl_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
        )
        inside_scl = scl[inside_aoi]
        scl_valid_fraction = float((~np.isin(inside_scl, list(SCL_EXCLUDED_CLASSES))).mean())
        scl_obscured = float((1.0 - scl_valid_fraction) * 100.0)
        scl = np.where(inside_aoi, scl, 0).astype(np.uint8)
    if require_cloud_threshold:
        if scl is None or scl_obscured is None:
            raise RuntimeError("strict curation requires a readable Sentinel SCL layer")
        if scl_obscured > max_cloud:
            raise RuntimeError(
                "selected Sentinel scene exceeds the AOI SCL obscured-pixel threshold "
                f"({scl_obscured:.3f}% > {max_cloud:.3f}%)"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    final_destination = output_root / aoi.aoi_id
    destination = Path(
        tempfile.mkdtemp(prefix=f".{aoi.aoi_id}.tmp-", dir=output_root)
    )
    sentinel_stack = np.stack([reflectance[name] for name in SENTINEL_BANDS])
    sentinel_output = np.where(np.isfinite(sentinel_stack), sentinel_stack, -9999.0).astype(
        np.float32
    )
    sentinel_profile = profile.copy()
    sentinel_profile.update(dtype="float32", nodata=-9999.0)
    sentinel_path = destination / "sentinel2_multispectral.tif"
    write_tif(sentinel_path, sentinel_output, sentinel_profile, SENTINEL_BANDS)
    Image.fromarray(
        robust_rgb_preview(reflectance["red"], reflectance["green"], reflectance["blue"])
    ).save(destination / "sentinel2_preview.png")

    ndvi, ndwi, ndbi = spectral_indices(
        reflectance["red"],
        reflectance["green"],
        reflectance["nir"],
        reflectance["swir16"],
    )
    if scl is not None:
        valid_scl = ~np.isin(scl, list(SCL_EXCLUDED_CLASSES))
        ndvi = np.where(valid_scl, ndvi, np.nan).astype(np.float32)
        ndwi = np.where(valid_scl, ndwi, np.nan).astype(np.float32)
        ndbi = np.where(valid_scl, ndbi, np.nan).astype(np.float32)
    indices = np.stack((ndvi, ndwi, ndbi)).astype(np.float32)
    index_output = np.where(np.isfinite(indices), indices, -9999.0).astype(np.float32)
    index_profile = profile.copy()
    index_profile.update(dtype="float32", nodata=-9999.0)
    write_tif(
        destination / "sentinel2_indices.tif",
        index_output,
        index_profile,
        ("ndvi", "ndwi", "ndbi"),
    )
    if scl is not None:
        scl_profile = profile.copy()
        scl_profile.update(dtype="uint8", nodata=0)
        write_tif(
            destination / "sentinel2_scl.tif",
            scl[None, ...],
            scl_profile,
            ("sentinel2_scene_classification",),
        )

    lon, lat = bbox_center(aoi.bbox)
    wc_href = worldcover_map_url(lon, lat)
    worldcover = reproject_single_band(
        wc_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
    )
    worldcover = np.where(inside_aoi, worldcover, 0).astype(np.uint8)
    label_profile = profile.copy()
    label_profile.update(count=1, dtype="uint8", nodata=0)
    write_tif(
        destination / "worldcover_2021_on_sentinel.tif",
        worldcover[None, ...],
        label_profile,
        ("worldcover_class",),
    )

    tree_href = hansen_url("treecover2000", lon, lat)
    loss_href = hansen_url("lossyear", lon, lat)
    treecover = reproject_single_band(
        tree_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
    )
    lossyear = reproject_single_band(
        loss_href, profile, dtype=np.dtype("uint8"), resampling=Resampling.nearest
    )
    treecover = np.where(inside_aoi, treecover, 0).astype(np.uint8)
    lossyear = np.where(inside_aoi, lossyear, 0).astype(np.uint8)
    write_tif(
        destination / "hansen_treecover2000_on_sentinel.tif",
        treecover[None, ...],
        label_profile,
        ("treecover2000_percent",),
    )
    write_tif(
        destination / "hansen_lossyear_on_sentinel.tif",
        lossyear[None, ...],
        label_profile,
        ("lossyear",),
    )

    output_files = [
        sentinel_path,
        destination / "sentinel2_indices.tif",
        destination / "sentinel2_preview.png",
        destination / "worldcover_2021_on_sentinel.tif",
        destination / "hansen_treecover2000_on_sentinel.tif",
        destination / "hansen_lossyear_on_sentinel.tif",
    ]
    if scl is not None:
        output_files.append(destination / "sentinel2_scl.tif")
    output_integrity = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in output_files
    }

    item_properties = item.get("properties", {})
    forest_mask = (treecover >= 30) & inside_aoi
    valid_tree = treecover[inside_aoi & (treecover <= 100)]
    manifest = {
        "schema_version": 4,
        "created_at": datetime.now(UTC).isoformat(),
        "integrity": {
            "builder": "scripts/prepare_geospatial_dataset.py",
            "max_pixels": max_pixels,
            "outputs": output_integrity,
        },
        "aoi": {
            "key": aoi.aoi_id,
            "name": aoi.name,
            "bbox": aoi.bbox,
            "purpose": aoi.purpose,
            "role": aoi.role.value,
            "geometry": aoi.geometry,
            "geometry_id": aoi.geometry_id,
        },
        "sentinel2": {
            "item_id": item.get("id"),
            "requested_start": start.isoformat() if start is not None else None,
            "requested_end": end.isoformat() if end is not None else None,
            "datetime": item_properties.get("datetime"),
            "cloud_cover": item_properties.get("eo:cloud_cover"),
            "cloud_threshold_required": require_cloud_threshold,
            "cloud_quality_basis": (
                "aoi_scl_obscured_percent" if require_cloud_threshold else "scene_metadata_rank"
            ),
            "aoi_scl_valid_fraction": (
                round(scl_valid_fraction, 6) if scl_valid_fraction is not None else None
            ),
            "aoi_scl_obscured_percent": (
                round(scl_obscured, 6) if scl_obscured is not None else None
            ),
            "collection": "sentinel-2-c1-l2a",
            "selection_policy": (
                "rank a bounded set of low-cloud candidates by STAC metadata, then require the "
                "actual AOI Sentinel SCL obscured-pixel fraction to meet the configured ceiling"
                if require_cloud_threshold
                else "prefer the lowest scene-wide STAC cloud estimate in the declared window"
            ),
            "bands": list(SENTINEL_BANDS),
            "band_cogs": band_hrefs,
            "band_calibration": {
                name: asset_calibration(assets[name]) for name in SENTINEL_BANDS
            },
            "scl_cog": scl_href,
            "scl_excluded_classes": sorted(SCL_EXCLUDED_CLASSES),
            "scl_valid_fraction": (
                round(scl_valid_fraction, 6) if scl_valid_fraction is not None else None
            ),
            "indices": {
                "ndvi": index_stats(ndvi),
                "ndwi": index_stats(ndwi),
                "ndbi": index_stats(ndbi),
            },
        },
        "worldcover": {
            "version": "2021 v200",
            "source": wc_href,
            "class_fractions": class_stats(worldcover),
        },
        "hansen": {
            "version": "GFC-2025-v1.13",
            "treecover2000_source": tree_href,
            "lossyear_source": loss_href,
            "mean_treecover2000_percent": (
                round(float(valid_tree.mean()), 4) if valid_tree.size else None
            ),
            "forest_pixels_ge_30pct": int(forest_mask.sum()),
            "loss_pixels_2001_2025": int(((lossyear > 0) & forest_mask).sum()),
        },
        "alignment": {
            "crs": str(profile["crs"]),
            "width": profile["width"],
            "height": profile["height"],
            "transform": list(profile["transform"])[:6],
            "note": (
                "All bands/references were reprojected onto the red-band Sentinel grid. "
                "Continuous Sentinel bands use bilinear resampling; categorical layers "
                "use nearest. "
                "Reflectance uses each STAC asset's raster:bands scale/offset metadata."
            ),
        },
        "scientific_warning": (
            "WorldCover is a reference land-cover product, not perfect ground truth; Hansen v1.13 "
            "has documented temporal-consistency limitations. Use spatial/temporal holdouts."
        ),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")

    backup = final_destination.with_name(f".{final_destination.name}.previous")
    try:
        if backup.exists():
            shutil.rmtree(backup)
        if final_destination.exists():
            os.replace(final_destination, backup)
        try:
            os.replace(destination, final_destination)
        except BaseException:
            if backup.exists() and not final_destination.exists():
                os.replace(backup, final_destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if destination.exists():
            shutil.rmtree(destination)
    return final_destination / "manifest.json"



def build_processing_cell_dataset(
    aoi: AOI,
    grid: ProcessingGrid,
    cell: ProcessingCell,
    output_root: Path,
    lookback_days: int,
    max_cloud: float,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cloud_threshold: bool = False,
    cells_root: Path | None = None,
    run_id: str | None = None,
    recipe_id: str | None = None,
    expected_source_selection_id: str | None = None,
) -> Path:
    """Build one independently executable processing-cell artifact.

    Collection execution supplies a frozen UTC window, run identity and recipe identity. Standalone
    callers remain supported for tests/tools, but collection mode is the reproducible path.
    """

    if cell not in grid.cells:
        raise ValueError("processing cell is not part of the supplied AOI grid")
    if (run_id is None) != (recipe_id is None):
        raise ValueError("run_id and recipe_id must be supplied together")
    if expected_source_selection_id is not None and (run_id is None or recipe_id is None):
        raise ValueError("source-selection continuity requires run and recipe identities")
    resumable_execution = run_id is not None
    if run_id is None:
        standalone_window = {
            "start": utc_iso(start) if start is not None else None,
            "end": utc_iso(end) if end is not None else None,
            "lookback_days": lookback_days,
        }
        run_id = canonical_sha256(
            {
                "mode": "standalone_nonresumable",
                "aoi_geometry_id": aoi.geometry_id,
                "cell_id": cell.cell_id,
                "window": standalone_window,
            }
        )
        recipe_id = cell_recipe_identity(run_id, cell.cell_id)
    assert recipe_id is not None
    target_geometry = cell_core_geometry(aoi, grid, cell)
    selected_items = stac_item_set(
        cell.wgs84_bbox,
        lookback_days,
        max_cloud,
        geometry=target_geometry,
        start=start,
        end=end,
        require_cloud_threshold=require_cloud_threshold,
        quality_max_pixels=min(grid.cell_pixels, 512),
    )
    target_profile = cell_profile(grid, cell, dtype="float32", nodata=-9999.0)
    inside_aoi = cell_aoi_mask(aoi, grid, cell, target_profile)

    reflectance: dict[str, np.ndarray] = {}
    band_sources: dict[str, list[dict[str, Any]]] = {}
    for band_name in SENTINEL_BANDS:
        calibrated_sources: list[tuple[str, float, float, float | None]] = []
        source_manifest: list[dict[str, Any]] = []
        for item in selected_items:
            assets = item.get("assets", {})
            if not isinstance(assets, dict):
                raise RuntimeError("selected Sentinel item has no valid asset map")
            asset = assets.get(band_name)
            if not isinstance(asset, dict):
                raise RuntimeError(f"selected Sentinel item is missing band {band_name}")
            href = validated_asset_href(asset.get("href"))
            calibration = asset_calibration(asset)
            nodata_value = calibration["nodata"]
            nodata = float(nodata_value) if isinstance(nodata_value, int | float) else None
            calibrated_sources.append(
                (
                    href,
                    float(calibration["scale"]),
                    float(calibration["offset"]),
                    nodata,
                )
            )
            source_manifest.append(
                {
                    "item_id": item.get("id"),
                    "href": href,
                    "calibration": calibration,
                }
            )
        band = reproject_calibrated_sources(calibrated_sources, target_profile)
        missing = inside_aoi & ~np.isfinite(band)
        if missing.any():
            raise RuntimeError(
                f"Sentinel band {band_name} does not fully cover the processing-cell target "
                f"({int(missing.sum())} pixels missing)"
            )
        reflectance[band_name] = np.where(inside_aoi, band, np.nan).astype(np.float32)
        band_sources[band_name] = source_manifest

    scl_hrefs: list[str] = []
    for item in selected_items:
        assets = item.get("assets", {})
        if isinstance(assets, dict):
            href = scl_asset_href(assets)
            if href is not None:
                scl_hrefs.append(href)
    scl: np.ndarray | None = None
    scl_valid_fraction: float | None = None
    scl_obscured: float | None = None
    if scl_hrefs:
        scl = reproject_sources(
            scl_hrefs,
            target_profile,
            dtype=np.dtype("uint8"),
            resampling=Resampling.nearest,
            destination_nodata=0,
        )
        inside_scl = scl[inside_aoi]
        scl_valid_fraction = float((~np.isin(inside_scl, list(SCL_EXCLUDED_CLASSES))).mean())
        scl_obscured = float((1.0 - scl_valid_fraction) * 100.0)
        scl = np.where(inside_aoi, scl, 0).astype(np.uint8)
    if require_cloud_threshold:
        if scl is None or scl_obscured is None:
            raise RuntimeError("strict cell curation requires readable Sentinel SCL coverage")
        if scl_obscured > max_cloud:
            raise RuntimeError(
                "assembled Sentinel cell exceeds the SCL obscured-pixel threshold "
                f"({scl_obscured:.3f}% > {max_cloud:.3f}%)"
            )

    worldcover_sources = [worldcover_map_url_for_tile(tile) for tile in cell.worldcover_tiles]
    hansen_tree_sources = [
        hansen_url_for_tile("treecover2000", tile) for tile in cell.hansen_tiles
    ]
    hansen_loss_sources = [hansen_url_for_tile("lossyear", tile) for tile in cell.hansen_tiles]
    worldcover, worldcover_coverage = reproject_sources_with_coverage(
        worldcover_sources,
        target_profile,
        dtype=np.dtype("uint8"),
        resampling=Resampling.nearest,
        destination_nodata=0,
    )
    treecover, treecover_coverage = reproject_sources_with_coverage(
        hansen_tree_sources,
        target_profile,
        dtype=np.dtype("uint8"),
        resampling=Resampling.nearest,
        destination_nodata=0,
    )
    lossyear, lossyear_coverage = reproject_sources_with_coverage(
        hansen_loss_sources,
        target_profile,
        dtype=np.dtype("uint8"),
        resampling=Resampling.nearest,
        destination_nodata=0,
    )
    for layer_name, coverage in (
        ("WorldCover", worldcover_coverage),
        ("Hansen treecover2000", treecover_coverage),
        ("Hansen lossyear", lossyear_coverage),
    ):
        missing = inside_aoi & ~coverage
        if missing.any():
            raise RuntimeError(
                f"{layer_name} sources do not fully cover the processing-cell target "
                f"({int(missing.sum())} pixels missing)"
            )
    worldcover = np.where(inside_aoi, worldcover, 0).astype(np.uint8)
    treecover = np.where(inside_aoi, treecover, 0).astype(np.uint8)
    lossyear = np.where(inside_aoi, lossyear, 0).astype(np.uint8)

    effective_cells_root = cells_root or (output_root / aoi.aoi_id / "cells")
    effective_cells_root.mkdir(parents=True, exist_ok=True)
    final_destination = effective_cells_root / cell.cell_id
    destination = Path(
        tempfile.mkdtemp(prefix=f".{cell.cell_id}.tmp-", dir=effective_cells_root)
    )
    try:
        sentinel_stack = np.stack([reflectance[name] for name in SENTINEL_BANDS])
        sentinel_output = np.where(np.isfinite(sentinel_stack), sentinel_stack, -9999.0).astype(
            np.float32
        )
        sentinel_profile = target_profile.copy()
        sentinel_profile.update(dtype="float32", nodata=-9999.0)
        sentinel_path = destination / "sentinel2_multispectral.tif"
        write_tif(sentinel_path, sentinel_output, sentinel_profile, SENTINEL_BANDS)
        Image.fromarray(
            robust_rgb_preview(reflectance["red"], reflectance["green"], reflectance["blue"])
        ).save(destination / "sentinel2_preview.png")

        ndvi, ndwi, ndbi = spectral_indices(
            reflectance["red"], reflectance["green"], reflectance["nir"], reflectance["swir16"]
        )
        if scl is not None:
            valid_scl = ~np.isin(scl, list(SCL_EXCLUDED_CLASSES))
            ndvi = np.where(valid_scl, ndvi, np.nan).astype(np.float32)
            ndwi = np.where(valid_scl, ndwi, np.nan).astype(np.float32)
            ndbi = np.where(valid_scl, ndbi, np.nan).astype(np.float32)
        index_output = np.where(
            np.isfinite(np.stack((ndvi, ndwi, ndbi))),
            np.stack((ndvi, ndwi, ndbi)),
            -9999.0,
        ).astype(np.float32)
        write_tif(
            destination / "sentinel2_indices.tif",
            index_output,
            sentinel_profile,
            ("ndvi", "ndwi", "ndbi"),
        )
        if scl is not None:
            categorical_profile = target_profile.copy()
            categorical_profile.update(dtype="uint8", nodata=0)
            write_tif(
                destination / "sentinel2_scl.tif",
                scl[None, ...],
                categorical_profile,
                ("sentinel2_scene_classification",),
            )
        categorical_profile = target_profile.copy()
        categorical_profile.update(dtype="uint8", nodata=0)
        write_tif(
            destination / "worldcover_2021_on_cell.tif",
            worldcover[None, ...],
            categorical_profile,
            ("worldcover_class",),
        )
        write_tif(
            destination / "hansen_treecover2000_on_cell.tif",
            treecover[None, ...],
            categorical_profile,
            ("treecover2000_percent",),
        )
        write_tif(
            destination / "hansen_lossyear_on_cell.tif",
            lossyear[None, ...],
            categorical_profile,
            ("lossyear",),
        )

        output_files = sorted(
            [path for path in destination.iterdir() if path.is_file()], key=lambda path: path.name
        )
        output_integrity = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        }
        forest_mask = (treecover >= 30) & inside_aoi
        valid_tree = treecover[inside_aoi & (treecover <= 100)]
        source_selection = {
            "sentinel2": {
                "item_ids": [item.get("id") for item in selected_items],
                "band_sources": band_sources,
                "scl_sources": scl_hrefs,
            },
            "worldcover": {"version": "2021 v200", "sources": worldcover_sources},
            "hansen": {
                "version": "GFC-2025-v1.13",
                "treecover2000_sources": hansen_tree_sources,
                "lossyear_sources": hansen_loss_sources,
            },
        }
        selection_id = source_selection_identity(source_selection)
        if (
            expected_source_selection_id is not None
            and selection_id != expected_source_selection_id
        ):
            raise RuntimeError(
                "rebuilding this processing cell would change its recorded source selection; "
                "start a fresh run instead"
            )

        manifest = {
            "schema_version": CELL_MANIFEST_SCHEMA_VERSION,
            "dataset_layout": CELL_DATASET_LAYOUT,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "recipe_id": recipe_id,
            "execution_mode": (
                "collection_resumable" if resumable_execution else "standalone_nonresumable"
            ),
            "source_selection_id": selection_id,
            "aoi": {
                "key": aoi.aoi_id,
                "name": aoi.name,
                "geometry_id": aoi.geometry_id,
                "role": aoi.role.value,
            },
            "cell": {
                **cell.as_dict(),
                "exact_geometry": target_geometry,
                "working_crs": grid.working_crs.definition,
                "working_crs_id": grid.working_crs.identity,
                "resolution_m": grid.resolution_m,
                "cell_pixels": grid.cell_pixels,
                "halo_pixels": grid.halo_pixels,
            },
            "sentinel2": {
                "item_ids": [item.get("id") for item in selected_items],
                "items": [
                    {
                        "id": item.get("id"),
                        "datetime": item.get("properties", {}).get("datetime"),
                        "eo_cloud_cover": item.get("properties", {}).get("eo:cloud_cover"),
                        "cell_scl_obscured_percent": item.get("properties", {}).get(
                            "gvo:cell_scl_obscured_percent"
                        ),
                    }
                    for item in selected_items
                ],
                "band_sources": band_sources,
                "scl_sources": scl_hrefs,
                "assembled_scl_valid_fraction": (
                    round(scl_valid_fraction, 6) if scl_valid_fraction is not None else None
                ),
                "assembled_scl_obscured_percent": (
                    round(scl_obscured, 6) if scl_obscured is not None else None
                ),
                "indices": {
                    "ndvi": index_stats(ndvi),
                    "ndwi": index_stats(ndwi),
                    "ndbi": index_stats(ndbi),
                },
            },
            "worldcover": {
                "version": "2021 v200",
                "sources": worldcover_sources,
                "class_fractions": class_stats(worldcover),
            },
            "hansen": {
                "version": "GFC-2025-v1.13",
                "treecover2000_sources": hansen_tree_sources,
                "lossyear_sources": hansen_loss_sources,
                "mean_treecover2000_percent": (
                    round(float(valid_tree.mean()), 4) if valid_tree.size else None
                ),
                "forest_pixels_ge_30pct": int(forest_mask.sum()),
                "loss_pixels_2001_2025": int(((lossyear > 0) & forest_mask).sum()),
            },
            "integrity": {"outputs": output_integrity},
            "scientific_warning": (
                "This cell is one part of a logical AOI. Scientific split identity is inherited "
                "from the AOI; cells are not independent train/validation/test evidence."
            ),
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        backup = final_destination.with_name(f".{final_destination.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if final_destination.exists():
            os.replace(final_destination, backup)
        try:
            os.replace(destination, final_destination)
        except BaseException:
            if backup.exists() and not final_destination.exists():
                os.replace(backup, final_destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return final_destination / "manifest.json"
    finally:
        if destination.exists():
            shutil.rmtree(destination)


def build_processing_cell_collection(
    aoi: AOI,
    output_root: Path,
    lookback_days: int,
    max_cloud: float,
    *,
    resolution_m: float = 10.0,
    cell_pixels: int = 1024,
    halo_pixels: int = 32,
    start: datetime | None = None,
    end: datetime | None = None,
    require_cloud_threshold: bool = False,
    force_rebuild: bool = False,
    fresh_window: bool = False,
) -> Path:
    """Build or resume one deterministic processing-cell run for an AOI.

    A dynamic lookback window is resolved once and persisted before the first cell runs. Re-entry
    with the same request resumes that exact window. Completed cells are reused only when their
    recipe identity and all recorded artifact hashes/sizes still validate.
    """

    if fresh_window and start is not None:
        raise ValueError("fresh_window is only meaningful for a dynamic lookback window")
    grid = plan_processing_grid(
        aoi,
        resolution_m=resolution_m,
        cell_pixels=cell_pixels,
        halo_pixels=halo_pixels,
    )
    grid_signature = canonical_sha256(grid.as_dict())
    request_id, request_payload = request_identity(
        aoi_id=aoi.aoi_id,
        geometry_id=aoi.geometry_id,
        grid_signature=grid_signature,
        lookback_days=lookback_days,
        max_cloud=max_cloud,
        require_cloud_threshold=require_cloud_threshold,
        explicit_start=start,
        explicit_end=end,
        pipeline_version=__version__,
    )

    aoi_root = output_root / aoi.aoi_id
    aoi_root.mkdir(parents=True, exist_ok=True)
    state_path = aoi_root / "run-state.json"
    existing_state = load_json_object(state_path)
    resumable_state = (
        not fresh_window
        and isinstance(existing_state, dict)
        and existing_state.get("schema_version") == RUN_STATE_SCHEMA_VERSION
        and existing_state.get("request_id") == request_id
        and existing_state.get("status") in {"running", "complete"}
    )

    resolved_start: datetime
    resolved_end: datetime
    resolved_run_id: str
    started_at: str
    if resumable_state:
        try:
            resolved_start = parse_utc(existing_state.get("resolved_start"), label="resolved_start")
            resolved_end = parse_utc(existing_state.get("resolved_end"), label="resolved_end")
            candidate_run_id = existing_state.get("run_id")
            if not isinstance(candidate_run_id, str) or len(candidate_run_id) != 64:
                raise ValueError("run_id is invalid")
            expected_run_id = run_identity(
                request_id, resolved_start=resolved_start, resolved_end=resolved_end
            )
            if candidate_run_id != expected_run_id:
                raise ValueError("run-state identity does not match its frozen acquisition window")
            resolved_run_id = candidate_run_id
            prior_started_at = existing_state.get("started_at")
            started_at = prior_started_at if isinstance(prior_started_at, str) else utc_iso(datetime.now(UTC))
        except ValueError:
            resumable_state = False

    if not resumable_state:
        resolved_start, resolved_end = resolve_run_window(
            lookback_days, start=start, end=end
        )
        resolved_run_id = run_identity(
            request_id, resolved_start=resolved_start, resolved_end=resolved_end
        )
        started_at = utc_iso(datetime.now(UTC))

    run_root = aoi_root / "runs" / resolved_run_id
    cells_root = run_root / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "status": "running",
        "pipeline_version": __version__,
        "request_id": request_id,
        "request": request_payload,
        "run_id": resolved_run_id,
        "aoi": {
            "key": aoi.aoi_id,
            "geometry_id": aoi.geometry_id,
            "role": aoi.role.value,
        },
        "grid_signature": grid_signature,
        "resolved_start": utc_iso(resolved_start),
        "resolved_end": utc_iso(resolved_end),
        "started_at": started_at,
        "updated_at": utc_iso(datetime.now(UTC)),
        "total_cells": len(grid.cells),
        "completed_cells": 0,
        "built_cells": 0,
        "reused_cells": 0,
    }
    atomic_write_json(state_path, state)

    cell_rows: list[dict[str, Any]] = []
    built_cells = 0
    reused_cells = 0
    try:
        for index, cell in enumerate(grid.cells, start=1):
            recipe_id = cell_recipe_identity(resolved_run_id, cell.cell_id)
            manifest_path = cells_root / cell.cell_id / "manifest.json"
            valid_manifest = None
            if not force_rebuild:
                valid_manifest = validate_processing_cell_manifest(
                    manifest_path,
                    expected_cell_id=cell.cell_id,
                    expected_recipe_id=recipe_id,
                    expected_run_id=resolved_run_id,
                )

            if valid_manifest is not None:
                reused_cells += 1
                disposition = "reused"
            else:
                previous_selection_id = prior_source_selection_id(
                    manifest_path,
                    expected_cell_id=cell.cell_id,
                    expected_recipe_id=recipe_id,
                )
                manifest_path = build_processing_cell_dataset(
                    aoi,
                    grid,
                    cell,
                    output_root,
                    lookback_days,
                    max_cloud,
                    start=resolved_start,
                    end=resolved_end,
                    require_cloud_threshold=require_cloud_threshold,
                    cells_root=cells_root,
                    run_id=resolved_run_id,
                    recipe_id=recipe_id,
                    expected_source_selection_id=previous_selection_id,
                )
                valid_manifest = validate_processing_cell_manifest(
                    manifest_path,
                    expected_cell_id=cell.cell_id,
                    expected_recipe_id=recipe_id,
                    expected_run_id=resolved_run_id,
                )
                if valid_manifest is None:
                    raise RuntimeError(
                        f"new processing-cell artifact failed integrity validation: {cell.cell_id}"
                    )
                built_cells += 1
                disposition = "built"

            cell_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "recipe_id": recipe_id,
                    "disposition": disposition,
                    "manifest": str(manifest_path.relative_to(aoi_root)).replace("\\", "/"),
                    "manifest_sha256": sha256_file(manifest_path),
                    "source_selection_id": valid_manifest["source_selection_id"],
                }
            )
            state.update(
                {
                    "updated_at": utc_iso(datetime.now(UTC)),
                    "completed_cells": index,
                    "built_cells": built_cells,
                    "reused_cells": reused_cells,
                }
            )
            state.pop("last_error", None)
            atomic_write_json(state_path, state)
    except BaseException as error:
        state.update(
            {
                "updated_at": utc_iso(datetime.now(UTC)),
                "completed_cells": len(cell_rows),
                "built_cells": built_cells,
                "reused_cells": reused_cells,
                "last_error": {
                    "type": type(error).__name__,
                    "message": str(error)[:1000],
                },
            }
        )
        atomic_write_json(state_path, state)
        raise

    collection = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "dataset_layout": COLLECTION_DATASET_LAYOUT,
        "pipeline_version": __version__,
        "run_id": resolved_run_id,
        "request_id": request_id,
        "resolved_window": {
            "start": utc_iso(resolved_start),
            "end": utc_iso(resolved_end),
        },
        "aoi": {
            "key": aoi.aoi_id,
            "name": aoi.name,
            "geometry_id": aoi.geometry_id,
            "role": aoi.role.value,
            "geometry": aoi.geometry,
        },
        "processing_grid": grid.as_dict(),
        "grid_signature": grid_signature,
        "cells": cell_rows,
        "execution": {
            "total_cells": len(cell_rows),
            "built_cells": built_cells,
            "reused_cells": reused_cells,
            "force_rebuild": bool(force_rebuild),
        },
        "scientific_unit": "AOI",
        "note": (
            "Cells are deterministic execution/artifact units. They inherit the AOI scientific "
            "role and must not be treated as independent evidence splits."
        ),
    }
    destination = run_root / "cells-manifest.json"
    atomic_write_json(destination, collection)
    state.update(
        {
            "status": "complete",
            "updated_at": utc_iso(datetime.now(UTC)),
            "completed_cells": len(cell_rows),
            "built_cells": built_cells,
            "reused_cells": reused_cells,
            "collection_manifest": str(destination.relative_to(aoi_root)).replace("\\", "/"),
            "collection_manifest_sha256": sha256_file(destination),
        }
    )
    state.pop("last_error", None)
    atomic_write_json(state_path, state)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", choices=sorted(CURATED_AOIS), default=None)
    parser.add_argument(
        "--aoi-file",
        type=Path,
        help="AOI v2 GeoJSON Feature; mutually exclusive with --aoi",
    )
    parser.add_argument("--output", type=Path, default=Path("data/curated"))
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help=(
            "maximum AOI SCL obscured-pixel percentage in strict mode; scene-wide "
            "eo:cloud_cover is used only to rank discovery candidates"
        ),
    )
    parser.add_argument(
        "--require-cloud-threshold",
        action="store_true",
        help="require the AOI SCL obscured-pixel fraction to satisfy --max-cloud",
    )
    parser.add_argument("--max-pixels", type=int, default=1024, choices=range(256, 2049))
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="optional UTC start date (YYYY-MM-DD); use with --end-date",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="optional UTC end date (YYYY-MM-DD); use with --start-date",
    )
    args = parser.parse_args()
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be provided together")
    if not 1 <= args.lookback_days <= 3650:
        parser.error("--lookback-days must be between 1 and 3650")
    if not math.isfinite(args.max_cloud) or not 0.0 <= args.max_cloud <= 100.0:
        parser.error("--max-cloud must be a finite percentage between 0 and 100")
    start = (
        datetime.combine(args.start_date, time.min, tzinfo=UTC)
        if args.start_date is not None
        else None
    )
    end = (
        datetime.combine(args.end_date, time.max, tzinfo=UTC)
        if args.end_date is not None
        else None
    )
    if args.aoi is not None and args.aoi_file is not None:
        parser.error("--aoi and --aoi-file are mutually exclusive")
    aoi_spec: AOI | None = None
    aoi_key = args.aoi or "helsinki_metro"
    if args.aoi_file is not None:
        try:
            payload = json.loads(args.aoi_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"unable to read --aoi-file: {error}")
        if not isinstance(payload, dict):
            parser.error("--aoi-file must contain one GeoJSON Feature object")
        try:
            aoi_spec = aoi_from_feature(payload)
        except ValueError as error:
            parser.error(str(error))
        aoi_key = aoi_spec.aoi_id
    path = build_dataset(
        aoi_key,
        args.output,
        args.lookback_days,
        args.max_cloud,
        aoi_spec=aoi_spec,
        max_pixels=args.max_pixels,
        start=start,
        end=end,
        require_cloud_threshold=args.require_cloud_threshold,
    )
    print(path)


if __name__ == "__main__":
    main()
